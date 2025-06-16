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
from utils.loss_MiDas import *
#from utils.loss_MiDas import MaskedScaleShiftInvariantLoss
#from data.VKITTI import KITTIVideoDataset
from data.Google_Landmark import GoogleLandmarksDataset, CombinedDataset
from video_depth_anything.video_depth import VideoDepthAnything
from benchmark.eval.metric import *
from benchmark.eval.eval_tae import tae_torch
from PIL import Image
from data.dataLoader import *

matplotlib.use('Agg')

MAX_DEPTH=80.0
CLIP_LEN =16

def least_sqaure_whole_clip(infs,gts):
    
    # 채널 없애고 [B,T,H,W] 로 만들기
    if infs.dim() == 5 and infs.shape[2] == 1:
        infs = infs.squeeze(2)
    if gts.dim() == 5 and gts.shape[2] == 1:
        gts  = gts .squeeze(2)

    ### 1. preprocessing
    valid_mask = (gts > 1e-3) & (gts < MAX_DEPTH)
    
    gt_disp_masked = 1. / (gts[valid_mask].reshape((-1,1)).double() + 1e-6)
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
    aligned_pred = torch.clamp(aligned_pred, min=1e-3)  ## 근데 왜 1/80 아니고 .,,,???
    #aligned_pred = aligned_pred[valid_mask].view(-1, 1).double()  # [N, 1] 꼴로 맞추기
    
    #print("scale : ",scale)
    #print("shift : ",shift)
    
    #print("aligned_pred.shape : ",aligned_pred.shape)    ## 예상대로라면, [~,1] 느낌이어야함 아 이게 아니네

    depth = torch.zeros_like(aligned_pred)
    depth = 1.0 / aligned_pred
    
    pred_depth = depth
    
    return pred_depth, valid_mask



def metric_val(infs, gts, poses, Ks):
    
    gt_depth = gts
    pred_depth, valid_mask = least_sqaure_whole_clip(infs,gts)
    
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
    #print("len_pred_depth : ",len(pred_depth))
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
    
    
def get_mask(depth_m, min_depth, max_depth):
    valid_mask = (depth_m > min_depth) & (depth_m < max_depth)
    return valid_mask.bool()


def norm_ssi(depth, valid_mask):
    
    eps=1e-6
    disparity = torch.zeros_like(depth)
    disparity[valid_mask] = 1.0 / depth[valid_mask]

    # 이거 마스크 씌우면 자동으로 펼쳐지니까 일단 내가 shape 가져가기
    B, T, C, H, W = disparity.shape
    disp_flat = disparity.view(B, T, -1)         # [B, T, H*W]
    mask_flat = valid_mask.view(B, T, -1)       # [B, T, H*W]

    # 마스크 빼고 민맥스 값 찾기
    disp_min = disp_flat.masked_fill(~mask_flat, float('inf')).min(dim=-1)[0]
    disp_max = disp_flat.masked_fill(~mask_flat, float('-inf')).max(dim=-1)[0]

    disp_min = disp_min.view(B, T, 1, 1, 1)
    disp_max = disp_max.view(B, T, 1, 1, 1)

    denom = (disp_max - disp_min + eps)
    norm_disp = (disparity - disp_min) / denom

    # 걍 invalid는 0으로 만들기
    norm_disp = norm_disp.masked_fill(~valid_mask, 0.0)

    return norm_disp
    
    

def train(args):

    ### 0. prepare GPU, wandb_login
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=api_key)


    ### 1. Handling hyper_params with WAND :)
    config_path = "configs/config.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    hyper_params = config["hyper_parameter"]
    
    run = wandb.init(project="Temporal_Diff_Flow_New_SSI", entity="Depth-Finder", config=hyper_params)

    lr = hyper_params["learning_rate"]
    #ratio_ssi = hyper_params["ratio_ssi"]
    #ratio_tgm = hyper_params["ratio_tgm"]
    #ratio_ssi_image = hyper_params["ratio_ssi_image"]
    num_epochs = hyper_params["epochs"]
    patient = hyper_params["patient"]
    batch_size = hyper_params["batch_size"]
    conv_out_channel = hyper_params["conv_out_channel"]
    conv = hyper_params["conv"]
    
    ratio_ssi       = args.ratio_ssi
    ratio_tgm       = args.ratio_tgm
    ratio_ssi_image = args.ratio_ssi_image
    threshold = args.static_th
    exp_name        = args.exp_name
    


    ### 2. Load data

    kitti_path = "/workspace/Video-Depth-Anything/datasets/KITTI"
    google_path="/workspace/Video-Depth-Anything/datasets/google_landmarks"

    rgb_clips, depth_clips = get_data_list(
        root_dir=kitti_path,
        data_name="kitti",
        split="train",
        clip_len=CLIP_LEN
    )

    kitti_train = KITTIVideoDataset(
        rgb_clips=rgb_clips,
        depth_clips=depth_clips,
        resize_size=518,
        split="train"
    )

    val_rgb_clips, val_depth_clips, val_cam_ids, val_intrin_clips, val_extrin_clips = get_data_list(
        root_dir=kitti_path,
        data_name="kitti",
        split="val",
        clip_len=CLIP_LEN
    )

    kitti_val = KITTIVideoDataset(
        rgb_clips=val_rgb_clips,
        depth_clips=val_depth_clips,
        cam_ids=val_cam_ids,
        intrin_clips=val_intrin_clips,
        extrin_clips=val_extrin_clips,
        resize_size=518,
        split="val"
    )

    google_img_paths, google_depth_paths = get_data_list(
        root_dir=google_path,
        data_name="google",
        split="train"
    )

    google_train = GoogleDepthDataset(
        img_paths=google_img_paths,
        depth_paths=google_depth_paths,
        resize_size=518
    )

    train_dataset = CombinedDataset(kitti_train,google_train,ratio=32)   
                    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=6)
    val_loader   = DataLoader(kitti_val,   batch_size=batch_size, shuffle=False, num_workers=6)

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

    loss_tgm = LossTGMVector(static_th=threshold)
    loss_ssi = Loss_ssi()

    wandb.watch(model, log="all")

    best_val_loss = float('inf')
    best_epoch = 0
    trial = 0

    scaler = GradScaler()

    ### 4. train
    
    print("train_loader)_len ", len(train_loader))

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        model.train()

        for batch_idx, ((x, y),(x_image, y_image)) in tqdm(enumerate(train_loader)):
            
            ## 마스크 생성
            video_masks = get_mask(y,min_depth=0.001,max_depth=80.0)
            x, y = x.to(device), y.to(device)
            video_masks = video_masks.to(device)
            
            # point. image는 B랑 T랑 바꿔줘야 계산할때 연속하지 않다고 판단할 수 있을 듯
            x_image = x_image.permute(1,0,2,3,4)
            y_image = y_image.permute(1,0,2,3,4)
            
            img_masks = get_mask(y_image,min_depth=1/80.0, max_depth= 1000.0)
            x_image, y_image = x_image.to(device), y_image.to(device)
            img_masks = img_masks.to(device)
            
            #print("x.shape",x.shape) # [16, 1, 3, 518, 518]
            #print("y.shape",y.shape) # [16, 1, 1, 518, 518]
            #print("video_masks.shape",video_masks.shape) # [16, 1, 1, 518, 518]
            
            optimizer.zero_grad()
            with autocast():
                pred = model(x)  
                #masks_squeezed = video_masks.squeeze(2)  # [B, T, H, W]
                #print("pred: ", pred)
                #print("gt :", y)
                print("")
                print("video : pred.mean():", pred.mean().item())
                #print("image : pred.mean():", pred_image.mean().item())
                #print("valid_mask.sum():", masks_squeezed.sum().item())
                #print("y.sum():", y.sum().item())
                #if pred.sum()== 0:
                #print("pred_sum = 0, see GT : ",y[0][0])
                
                pred_image = model(x_image)
                print("image : pred.mean():", pred_image.mean().item())
            
                
                #print("pred.isnan().any():", torch.isnan(pred).any().item())
                #print("pred.isinf().any():", torch.isinf(pred).any().item())
                #if pred.sum()==0:
                #print("pred_sum = 0, see GT : ",y[0][0])

                # 유효 픽셀에만 곱하기
                #pred_image_masked = pred_image * mask_image
                #y_image_masked    = y_image    * mask_image

                #loss_tgm_val = loss_tgm(pred_masked, y_masked, masks_squeezed)
                
                # 이거 norm_ssi 갔다오면 disp로 줌 
                disp_normed = norm_ssi(y,video_masks)
                video_masks_squeezed = video_masks.squeeze(2)
                loss_ssi_value = loss_ssi(pred, disp_normed, video_masks_squeezed)   ## 어차피 5->4 는 loss에서 해줌 
                
                
                #disp_normed = norm_ssi(y,video_masks)
                #infs_fit,_ = least_sqaure_whole_clip(infs=pred,gts=y)    # 이렇게 나온 결과는 depth임 
                
                #print("pred: ", pred[0][0])
                #print("infs_fit: ", infs_fit[0][0])
                #print("y: ", y[0][0])
                loss_tgm_value = loss_tgm(pred, y, video_masks_squeezed)
                
                # =============== single img =================== # 
                img_disp_normed = norm_ssi(y_image,img_masks)
                img_masks_squeezed = img_masks.squeeze(2)
                loss_ssi_image_value = loss_ssi(pred_image, img_disp_normed, img_masks_squeezed)
                
                #if epoch < 5 : 
                #    loss = ratio_ssi * loss_ssi_value + ratio_ssi_image * loss_ssi_image_value
                
                #else :
                loss = ratio_tgm * loss_tgm_value + ratio_ssi * loss_ssi_value + ratio_ssi_image * loss_ssi_image_value
                #print(">> loss.shape:", loss.shape)
                #print("check ",loss_tgm_value,loss_ssi_value,loss_ssi_image_value)
                #loss = loss_tgm_val * 0.1 + loss_ssi_val
                #loss = loss_ssi_val 
            
            # 또는 스케일링 방식 사용: 유효한 픽셀 수로 정규화
            # valid_pixel_ratio = masks.sum() / (masks.shape[0] * masks.shape[1] * masks.shape[2] * masks.shape[3] * masks.shape[4])
            # loss = (ratio_tgm * loss_tgm_val + ratio_ssi * loss_ssi_val) / (valid_pixel_ratio + 1e-8)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            #loss.backward()
            #optimizer.step()
            
            epoch_loss += loss.item()

            if batch_idx % 5 == 0:
                print(f"Epoch [{epoch}], Batch [{batch_idx}], Loss: {loss.item():.4f}")
            
        avg_train_loss = epoch_loss / len(train_loader)
        
        scheduler.step()

        print(f"Epoch [{epoch}/{num_epochs}] Train Loss: {avg_train_loss:.4f}")        
        
        model.eval()
        val_loss = 0.0

        total_absrel = 0.0
        total_delta1 = 0.0
        total_tae = 0.0
        cnt_clip = 0

        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(val_loader)):
                x, y, extrinsics, intrinsics = batch
                x, y = x.to(device), y.to(device)
                
                pred = model(x)
                masks = get_mask(y, min_depth=0.001, max_depth=80.0)
                masks = masks.to(device).bool()             # [B, T, 1, H, W]

                
                disp_normed = norm_ssi(y, masks)
                masks= masks.squeeze(2)
                ssi_loss_value = loss_ssi(pred, disp_normed, masks)
                
                infs_fit,_ = least_sqaure_whole_clip(infs=pred, gts=y)
                #infs_fit = 1 / (pred + 1e-6)
                
                #print("infs: ", infs_fit[0][0])
                #print("y: ", y[0][0])
                
                tgm_loss_value = loss_tgm(infs_fit, y, masks)

                loss = ratio_ssi * ssi_loss_value + ratio_tgm * tgm_loss_value
                val_loss += loss.item()

                # 8) accumulate metrics
                B, T, H, W = pred.shape
                poses = extrinsics.to(device)
                Ks   = intrinsics.to(device)
                
                with torch.no_grad():
                    B, T, C, H, W = x.shape
                    save_dir = f"outputs/modified_ssi/{exp_name}/frames/test/epoch_{epoch}_batch_{batch_idx}"
                    os.makedirs(save_dir, exist_ok=True)

                    #with autocast():
                    pred = model(x)
                    # 이제 B=1 가정 → B 차원 제거
                    rgb_clip   = x[0]  # [T, 3, H, W]
                    disp_clip  = disp_normed[0]       # [T, 1, H, W]
                    mask_clip  = masks[0]      # [T, 1, H, W]
                    pred_clip  = infs_fit[0]       # [T, 1, H, W]
                    
                    for t in range(T):
                        """
                        # --- a) RGB 저장 ---
                        rgb_frame = rgb_clip[t]  # [3, H, W], 값 ∈ [0,1]
                        rgb_np = (rgb_frame.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)  # [H, W, 3]
                        rgb_pil = Image.fromarray(rgb_np)
                        rgb_out = os.path.join(save_dir, f"rgb_frame_{t:02d}.png")
                        rgb_pil.save(rgb_out)
                        
                        """

                        #disp_frame = 1.0 / (disp_clip[t, 0] + 1e-6)          # [H, W]
                        #print(disp_frame.min().item(), disp_frame.max().item())
                        #disp_frame = disp_frame.clamp_(max=0.05) 
                        #d_min, d_max = disp_frame[mask_clip[t]].min(), disp_frame[mask_clip[t]].max()
                        #disp_norm = (disp_frame - d_min) / (d_max - d_min + 1e-9)
                        
                        disp_np = (disp_clip[t,0].cpu().numpy() * 255.0).astype(np.uint8)  # [H, W]
                        disp_rgb_np = np.stack([disp_np] * 3, axis=-1)                # [H, W, 3]
                        disp_pil = Image.fromarray(disp_rgb_np)
                        disp_out = os.path.join(save_dir, f"disp_frame_{t:02d}.png")
                        disp_pil.save(disp_out)

                        # --- c) Mask 저장 ---
                        mask_frame = mask_clip[t]  # [1, H, W], 값 ∈ {0, 255}
                        mask_np = (mask_frame.squeeze(0).cpu().numpy() * 255.0).astype(np.uint8)  # [H, W]
                        mask_pil = Image.fromarray(mask_np)
                        mask_out = os.path.join(save_dir, f"mask_frame_{t:02d}.png")
                        mask_pil.save(mask_out)
                        
                        # --- e) Prediction 저장 ---
                        # pred_clip[t] : [H, W], float16/float32 (raw 예측값, 스케일 미정)
                        pred_frame = pred_clip[t].cpu().float().numpy()  # → np.float32, shape=[H, W]
                        # per-frame 정규화 (min/max) → [0,1]
                        pmin, pmax = pred_frame.min(), pred_frame.max()
                        pred_norm = (pred_frame - pmin) / (pmax - pmin+1e-9)
                        pred_uint8 = (pred_norm * 255.0).astype(np.uint8)  # [H, W]
                        pred_rgb_np = np.stack([pred_uint8]*3, axis=-1)   # [H, W, 3]
                        pred_pil = Image.fromarray(pred_rgb_np)
                        pred_out = os.path.join(save_dir, f"pred_frame_norm_{t:02d}.png")
                        pred_pil.save(pred_out)
                        
                        if t == 0 :
                            break

                    print(f"  → Epoch {epoch}, Batch {batch_idx} 저장: '{save_dir}'")

                
                for b in range(B):
                    inf_clip   = pred[b]         # [clip_len, H, W]
                    gt_clip    = y[b].squeeze(1)            
                    poses_clip = poses[b]      
                    Ks_clip    = Ks[b]          

                    absrel, delta1, tae = metric_val(
                        inf_clip, gt_clip, poses_clip, Ks_clip
                    )

                    total_absrel += absrel
                    total_delta1 += delta1
                    total_tae += tae
                    cnt_clip += 1   ## 이름 이슈인데, cnt_clip는 배치 내 프레임의 개수로 사용
                    
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

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    experiments = [

        {"ratio_ssi": 1.0, "static_th" : 0.05, "ratio_tgm": 10.0, "ratio_ssi_image": 0.5},
        #{"ratio_ssi": 1.0, "static_th" : 0.003, "ratio_tgm": 1.0, "ratio_ssi_image": 0.5},
        #{"ratio_ssi": 1.0, "static_th" : 0.005, "ratio_tgm": 1.0, "ratio_ssi_image": 0.5},
        #{"ratio_ssi": 1.0, "static_th" : 0.007, "ratio_tgm": 1.0, "ratio_ssi_image": 0.5},
        #{"ratio_ssi": 1.0, "static_th" : 0.01, "ratio_tgm": 1.0, "ratio_ssi_image": 0.5},
        #{"ratio_ssi": 1.0, "static_th" : 0.02, "ratio_tgm": 1.0, "ratio_ssi_image": 0.5},
    ]

    for exp in experiments:
        for trial in range(3):  # 각 실험마다 3회 반복
            # argparse 로 받은 기본 args 복사
            run_args = argparse.Namespace(**vars(args))

            # 실험별 파라미터 덮어쓰기
            run_args.ratio_ssi       = exp["ratio_ssi"]
            run_args.ratio_tgm       = exp["ratio_tgm"]
            run_args.ratio_ssi_image = exp["ratio_ssi_image"]
            run_args.trial           = trial
            run_args.static_th = exp["static_th"]

            # exp_name에 trial 번호 포함
            run_args.exp_name = (
                f"ssi{run_args.ratio_ssi}"
                f"_tgm{run_args.ratio_tgm}"
                f"_staticTH_{run_args.static_th}"
                f"_ssiimg{run_args.ratio_ssi_image}"
                f"_trial{trial}"
            )

            print(f"\n===== Starting experiment: {run_args.exp_name} =====")
            train(run_args)