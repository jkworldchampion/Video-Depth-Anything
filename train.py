#! /usr/bin/python3
import os
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml
import wandb
import gc


from torch.utils.data import DataLoader 
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from utils.loss_MiDas import Loss_ssi, Loss_tgm
#from utils.loss_MiDas import MaskedScaleShiftInvariantLoss
from data.VKITTI import KITTIVideoDataset
from data.Google_Landmark import GoogleLandmarksDataset, CombinedDataset
from video_depth_anything.video_depth import VideoDepthAnything
from benchmark.eval.metric import *
from benchmark.eval.eval_tae import tae_torch
from PIL import Image

matplotlib.use('Agg')

MAX_DEPTH=80.0

def metric_val(infs, gts, poses, Ks):

    """
    least square 때문에, i & i+1 계산보다는 클립 하나를 통째로 계산하는 것이 올바름 
    infs,gts : [clip_len, ~] 꼴을 기대, inf는 

    지금 문제 : gt가 여기서는 진짜 gt를 기대하고있음 clipping 하기 전 ..
    
    LiheYoung씨가 rel-model인 경우에는 relative depth로 metric 해도 된다고 했음 
    -> 일단 gt말고 disparity랑 least square 때리기

    """
    
    ### 1. preprocessing
    #valid_mask = np.logical_and((gts>1e-3), (gts<MAX_DEPTH))
    valid_mask = (gts > 1e-3) & (gts < MAX_DEPTH)
    
    gt_disp_masked = 1. / (gts[valid_mask].reshape((-1,1)).double() + 1e-8)
    infs = infs.clamp(min=1e-3)
    pred_disp_masked = infs[valid_mask].reshape((-1,1)).double()
    
    ### 2. least square
    
    #print("valid_mask shape ( expecting 32 x H x W ) : ",valid_mask.shape)   
    #print("num valid pixels : ",valid_mask.sum())    ## 32개의 합이겠다. 
    
    #print("pred_disp_masked : ",pred_disp_masked)
    #print("pred_disp_masked.shape : ",pred_disp_masked.shape)
    #print("disparity_gts_disp_masked : ",disparity_gts_disp_masked)
    #print("disparity_gts_disp_masked.shape : ",disparity_gts_disp_masked.shape)
    
    _ones = torch.ones_like(pred_disp_masked)
    A = torch.cat([pred_disp_masked, _ones], dim=-1) 
    X = torch.linalg.lstsq(A, gt_disp_masked).solution  
    scale = X[0].item()
    shift = X[1].item()
    aligned_pred = scale * infs + shift
    aligned_pred = torch.clamp(aligned_pred, min=1e-3) 
    #aligned_pred = aligned_pred[valid_mask].view(-1, 1).double()  # [N, 1] 꼴로 맞추기
    
    #print("scale : ",scale)
    #print("shift : ",shift)
    
    #print("aligned_pred.shape : ",aligned_pred.shape)    ## 예상대로라면, [~,1] 느낌이어야함 아 이게 아니네

    depth = torch.zeros_like(aligned_pred)
    depth = 1.0 / aligned_pred
    
    gt_depth = gts
    #pred_depth = torch.clamp(depth, min=1e-3, max=MAX_DEPTH)
    
    pred_depth = depth

    ### 4. validity
    n = valid_mask.sum((-1, -2))    ## 인풋이 3차원이었으므로 T차원임
    #print("n.shape: ", n.shape)
    valid_frame = (n > 0)   # 어 이거 torch.Size([32, 518, 518]) -> 32, ? 꼴인데 지금 ?
    #print("pred_depth.shape: ", pred_depth.shape)
    #print("valid_frame.shape : ",valid_frame.shape)
    pred_depth = pred_depth[valid_frame] # ok 가능함
    gt_depth = gt_depth[valid_frame]
    valid_mask = valid_mask[valid_frame]
    
    #print("valid_mask : ",valid_mask)
    #print("valid_frame", valid_frame)
    
    absrel = abs_relative_difference(pred_depth, gt_depth, valid_mask)
    delta1 = delta1_acc(pred_depth, gt_depth, valid_mask)
    tae = eval_tae(pred_depth, gt_depth, poses, Ks, valid_mask)

    
    return absrel,delta1,tae


def eval_tae(pred_depth, gt_depth, poses, Ks, masks):
    
    error_sum = 0.
    print("len_pred_depth : ",len(pred_depth))
    for i in range(len(pred_depth) - 1):
        depth1 = pred_depth[i]
        depth2 = pred_depth[i+1]
        
        mask1 = masks[i]
        mask2 = masks[i+1]

        T_1 = poses[i]
        T_2 = poses[i+1]

        T_2_1 = torch.linalg.inv(T_2) @ T_1
   
        R_2_1 = T_2_1[:3,:3]
        t_2_1 = T_2_1[:3, 3]
        K = Ks[i]


        error1 = tae_torch(depth1, depth2, R_2_1, t_2_1, K, mask2)
        T_1_2 = torch.linalg.inv(T_2_1)
        R_1_2 = T_1_2[:3,:3]
        t_1_2 = T_1_2[:3, 3]


        error2 = tae_torch(depth2, depth1, R_1_2, t_1_2, K, mask1)
        
        error_sum += error1
        error_sum += error2
    
    result = error_sum / (2 * (len(pred_depth) -1))
    return result
    
    

def train():

    ### 0. prepare GPU, wandb_login
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=api_key)


    ### 1. Handling hyper_params with WAND :)
    config_path = "configs/config.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    hyper_params = config["hyper_parameter"]
    
    run = wandb.init(project="Temporal_Diff_Flow", entity="Depth-Finder", config=hyper_params)

    lr = hyper_params["learning_rate"]
    ratio_ssi = hyper_params["ratio_ssi"]
    ratio_tgm = hyper_params["ratio_tgm"]
    ratio_ssi_image = hyper_params["ratio_ssi_image"]
    num_epochs = hyper_params["epochs"]
    patient = hyper_params["patient"]
    batch_size = hyper_params["batch_size"]
    conv_out_channel = hyper_params["conv_out_channel"]
    conv = hyper_params["conv"]
    CLIP_LEN = hyper_params["clip_len"]


    ### 2. Load data

    kitti_path = "/workspace/Video-Depth-Anything/datasets/KITTI"

    # 2) 학습/검증 데이터셋 생성
    kitti_train = KITTIVideoDataset(
        root_dir=kitti_path,
        clip_len=CLIP_LEN,
        resize_size=518,
        split="train"
    )
    kitti_val = KITTIVideoDataset(
        root_dir=kitti_path,
        clip_len=CLIP_LEN,
        resize_size=518,
        split="val"
    )

    train_dataset = CombinedDataset(
        kitti_train,
        google_image_root="/workspace/Video-Depth-Anything/datasets/google_landmarks/images",
        google_depth_root="/workspace/Video-Depth-Anything/datasets/google_landmarks/depth",
        #output_size=518
    )
    val_dataset = CombinedDataset(
        kitti_val,
        google_image_root="/workspace/Video-Depth-Anything/datasets/google_landmarks/images",
        google_depth_root="/workspace/Video-Depth-Anything/datasets/google_landmarks/depth",
        #output_size=518
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)



    ### 3. Model and additional stuffs,...

    model = VideoDepthAnything(num_frames=CLIP_LEN,out_channels=[48, 96, 192, 384],out_channel=conv_out_channel,conv=conv).to(device)
    """
    out_channel : result of diff-conv
    conv : usage. False to use raw RGB diff
    """
    
    # 2) 체크포인트 파일 로드
    #model.load_state_dict(torch.load(f'video_depth_anything_vits.pth', map_location='cpu'), strict=False)
    #model = model.to(device).eval()
        
    # freeze -> pretrain은 DINO밖에 없어서 이렇게 가능 
    for param in model.pretrained.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters:", total_params)

    loss_tgm = Loss_tgm()
    loss_ssi = Loss_ssi()
    #loss_ssi = MaskedScaleShiftInvariantLoss()  # 이건 MiDaS 버전

    wandb.watch(model, log="all")

    best_val_loss = float('inf')
    best_epoch = 0
    trial = 0

    scaler = GradScaler()

    ### 4. train

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        model.train()

        for batch_idx, (x, y, masks, x_image, y_image, mask_image) in tqdm(enumerate(train_loader)):
            x, y = x.to(device), y.to(device)
            masks = masks.bool()
            masks = masks.to(device)
            
            x_image    = x_image.to(device)
            y_image    = y_image.to(device)
            mask_image = mask_image.to(device)
            
            if x_image.dim() == 4: # [B, C, H, W]
                x_image = x_image.unsqueeze(1) # [B, T, C, H, W]
            if y_image.dim() == 3:  # [B, H, W]
                y_image = y_image.unsqueeze(1) # [B, T, H, W]
            if mask_image.dim() == 3:   # [B, H, W]
                mask_image = mask_image.unsqueeze(1)  # [B, T, H, W]
                
            #print("x.shape",x.shape)
            #print("y.shape",y.shape)
            #print("masks.shape",masks.shape)
            
            #print("x:", x)  # [B, T, C, H, W]
            #print("y:", y)  # [B, T, 1, H, W]
            #print("masks:", masks)  # [B, T, 1, H, W]
            if y.dim() == 5:
                y = y[:, :, 0, :, :]  # now y.shape == [B, T, H, W]
            
            optimizer.zero_grad()
            with autocast():
                pred = model(x)  # pred.shape == [B, T, H, W]
                # masks.shape == [B, T, 1, H, W]
                # x_image: [B, C, H, W] → [B, 1, C, H, W] 로 차원 확장
                # if x_image.dim() == 4:
                #     x_image = x_image.unsqueeze(1)
                #pred_image = model(x_image)         # 이제 [B, 1, H, W] 형태로 나옵니다.
                        
                # 마스크 채널 축 제거
                masks_squeezed = masks.squeeze(2)  # [B, T, H, W]
                
                #print("pred: ", pred)
                #print("gt :", y)
                print("")
                print("video : pred.mean():", pred.mean().item())
                #print("image : pred.mean():", pred_image.mean().item())
                #print("valid_mask.sum():", masks_squeezed.sum().item())
                #print("y.sum():", y.sum().item())
                #if pred.sum()== 0:
                #print("pred_sum = 0, see GT : ",y[0][0])
            
                
                #print("pred.isnan().any():", torch.isnan(pred).any().item())
                #print("pred.isinf().any():", torch.isinf(pred).any().item())
                #if pred.sum()==0:
                #print("pred_sum = 0, see GT : ",y[0][0])

                # 유효 픽셀에만 곱하기
                pred_masked = pred * masks_squeezed
                y_masked    = y    * masks_squeezed
                
                pred_image_masked = pred_image * mask_image
                y_image_masked    = y_image    * mask_image

                #loss_tgm_val = loss_tgm(pred_masked, y_masked, masks_squeezed)
                loss_ssi_val = loss_ssi(pred_masked, y_masked, masks_squeezed)
                
                #print("pred_image_masked.shape:", pred_image_masked.shape)
                #print("y_image_masked.shape:", y_image_masked.shape)
                #print("pred_image", pred_image)
                #print("single image : pred.mean():", pred_image.mean().item())
                #print("y_image_", y_image)
                #print("y_image.mean()", y_image.mean().item())
                
                #loss_ssi_val_image = loss_ssi(pred_image_masked, y_image_masked, mask_image)
                
                #loss = ratio_tgm * loss_tgm_val + ratio_ssi * loss_ssi_val + ratio_ssi_image * loss_ssi_val_image
                loss = loss_ssi_val 
            
            # 또는 스케일링 방식 사용: 유효한 픽셀 수로 정규화
            # valid_pixel_ratio = masks.sum() / (masks.shape[0] * masks.shape[1] * masks.shape[2] * masks.shape[3] * masks.shape[4])
            # loss = (ratio_tgm * loss_tgm_val + ratio_ssi * loss_ssi_val) / (valid_pixel_ratio + 1e-8)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            #loss.backward()
            #optimizer.step()
            
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        scheduler.step()

        print(f"Epoch [{epoch}/{num_epochs}] Train Loss: {epoch_loss:.4f}")
        
        model.eval()
        val_loss = 0.0

        total_absrel = 0.0
        total_delta1 = 0.0
        total_tae = 0.0
        cnt_clip = 0

        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(val_loader)):
                x, y, masks, true_depth, extrinsics, intrinsics = batch
                x, y,true_depth = x.to(device), y.to(device),  true_depth.to(device)
                pred = model(x)
                masks = masks.bool()
                masks = masks.to(device)
                
                y = y[:, :, 0, :, :]

                ## 기억 안나서 그냥 추가 ;;...
                ## loss 구할때는 거기서 차원 맞춰줬었는데, 여기는 직접 해주기. b len 1 h w 에서 1 제거 

                if pred.dim() == 5 :
                    pred = pred.squeeze(2)

                B,clip_len,_,_ = pred.shape

                if y.dim() == 5 :
                    y = y.squeeze(2)

                if masks.dim() == 5 :
                    masks = masks.squeeze(2).bool()
                    
                if true_depth.dim() == 5:
                    true_depth = true_depth.squeeze(2)
                    
                #print("extrinsics :",extrinsics_list)
                #print("intrinsics :",intrinsics_list)
                
                poses = extrinsics.to(device)
                Ks = intrinsics.to(device)
                
                #print("true_depth:", true_depth[0])
                #print("true_depth.shape:", true_depth.shape)
                #print("disparity_gt:", y[0])
                #print("pred_depth : ",pred[0])
                #print("pred.sum(): ", pred.sum().item())

                with torch.no_grad():
                    B, T, C, H, W = x.shape
                    save_dir = f"outputs/test_train/without_tgm/frames/epoch_{epoch}_batch_{batch_idx}"
                    os.makedirs(save_dir, exist_ok=True)

                    #with autocast():
                    pred = model(x)
                    # 이제 B=1 가정 → B 차원 제거
                    rgb_clip   = x[0]  # [T, 3, H, W]
                    disp_clip  = y[0]       # [T, 1, H, W]
                    mask_clip  = masks[0]      # [T, 1, H, W]
                    pred_clip  = pred[0]       # [T, 1, H, W]
                    
                    for t in range(T):
                        # --- a) RGB 저장 ---
                        rgb_frame = rgb_clip[t]  # [3, H, W], 값 ∈ [0,1]
                        rgb_np = (rgb_frame.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)  # [H, W, 3]
                        rgb_pil = Image.fromarray(rgb_np)
                        rgb_out = os.path.join(save_dir, f"rgb_frame_{t:02d}.png")
                        rgb_pil.save(rgb_out)

                        # --- b) GT Disparity 저장 ---
                        disp_frame = disp_clip[t]  # [1, H, W], 값 ∈ [0,1]
                        disp_np = (disp_frame.squeeze(0).cpu().numpy() * 255.0).astype(np.uint8)  # [H, W]
                        disp_rgb_np = np.stack([disp_np]*3, axis=-1)  # [H, W, 3]
                        disp_pil = Image.fromarray(disp_rgb_np)
                        disp_out = os.path.join(save_dir, f"disp_frame_{t:02d}.png")
                        disp_pil.save(disp_out)

                        # --- c) Mask 저장 ---
                        mask_frame = mask_clip[t]  # [1, H, W], 값 ∈ {0, 255}
                        mask_np = (mask_frame.squeeze(0).cpu().numpy() * 255.0).astype(np.uint8)  # [H, W]
                        mask_pil = Image.fromarray(mask_np)
                        mask_out = os.path.join(save_dir, f"mask_frame_{t:02d}.png")
                        mask_pil.save(mask_out)

                        # --- d) Prediction 저장 ---
                        # pred_clip[t] : [H, W], float16/float32 (raw 예측값, 스케일 미정)
                        pred_frame = pred_clip[t].cpu().float().numpy()  # → np.float32, shape=[H, W]
                        # per-frame 정규화 (min/max) → [0,1]
                        pmin, pmax = pred_frame.min(), pred_frame.max()
                        pred_norm = (pred_frame - pmin) / (pmax - pmin+1e-9)
                        pred_uint8 = (pred_norm * 255.0).astype(np.uint8)  # [H, W]
                        pred_rgb_np = np.stack([pred_uint8]*3, axis=-1)   # [H, W, 3]
                        pred_pil = Image.fromarray(pred_rgb_np)
                        pred_out = os.path.join(save_dir, f"pred_frame_{t:02d}.png")
                        pred_pil.save(pred_out)

                    print(f"  → Epoch {epoch}, Batch {batch_idx} 저장: '{save_dir}'")

                
                for b in range(B):
                    inf_clip   = pred[b]         # [clip_len, H, W]
                    gt_clip    = true_depth[b]           
                    mask_clip  = masks[b]      
                    poses_clip = poses[b]      
                    Ks_clip    = Ks[b]          

                    absrel, delta1, tae = metric_val(
                        inf_clip, gt_clip, poses_clip, Ks_clip
                    )

                    total_absrel += absrel
                    total_delta1 += delta1
                    total_tae += tae
                    cnt_clip += 1   ## 이름 이슈인데, cnt_clip는 배치 내 프레임의 개수로 사용
                    
                # 검증 시에도 동일한 마스킹 적용
                masks_expanded = masks.expand_as(pred)
                pred_masked = pred * masks_expanded
                y_masked = y * masks_expanded
                masks_squeezed = masks.squeeze(2)
                
                # 손실 계산
                loss_tgm_val = loss_tgm(pred_masked, y_masked, masks_squeezed)
                loss_ssi_val = loss_ssi(pred_masked, y_masked, masks_squeezed)
                loss = ratio_tgm * loss_tgm_val + ratio_ssi * loss_ssi_val
                #loss = loss_ssi_val
                val_loss += loss.item()
                if batch_idx == 0:
                    break
                    
            avg_val_loss = val_loss / len(val_loader)
        
            avg_absrel = total_absrel / cnt_clip
            avg_delta1 = total_delta1 / cnt_clip
            avg_tae = total_tae / cnt_clip
            

        print(f"Epoch [{epoch}/{num_epochs}] Validation Loss: {avg_val_loss:.4f}")
        print(f"AbsRel  : {avg_absrel:.4f}")
        print(f"Delta1  : {avg_delta1:.4f}")
        print(f"TAE    : {avg_tae:.4f}")

        wandb.log({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "absrel": avg_absrel,
            "delta1": avg_delta1,
            "tae": avg_tae,
            "epoch": epoch
        })
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),    
                'scaler_state_dict': scaler.state_dict(),
            }, 'best_checkpoint.pth')
            print(f"Best checkpoint saved at epoch {epoch} with validation loss {avg_val_loss:.4f}")
            trial = 0
        else:
            trial += 1
            
        """
        if trial >= patient:
            print("Early stopping triggered.")
            break
        """
    # 최종 모델 저장

    print(f"Training finished. Best checkpoint was from epoch {best_epoch} with validation loss {best_val_loss:.4f}.")
    run.finish()

if __name__ == "__main__":
    train()