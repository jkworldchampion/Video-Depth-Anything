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
from utils.loss import Loss_ssi, Loss_tgm
from data import KITTIVideoDataset
from video_depth_anything.video_depth import VideoDepthAnything
from benchmark.eval.metric import *
from benchmark.eval.eval_tae import tae_torch
from benchmark.eval.eval import depth2disparity

matplotlib.use('Agg')

MAX_DEPTH=80.0

def count_total_frames(video_infos):
    """
    video_infos: 비디오 정보 딕셔너리의 리스트
    → 각 폴더 안의 파일 개수를 모두 합산하여 반환
    """
    total = 0
    for info in video_infos:  # video_pairs 대신 video_infos 사용
        rgb_path = info['rgb_path']  # 딕셔너리에서 키 접근
        files = [f for f in os.listdir(rgb_path) 
                 if os.path.isfile(os.path.join(rgb_path, f)) 
                 and (f.lower().endswith(".png") or f.lower().endswith(".jpg"))]
        total += len(files)
    return total

def test_vkitti_dataloader_fullcount():
    # ------------------------------------------------------------------------
    # 1) 데이터셋 루트 경로 (환경에 맞게 수정)
    kitti_path = "/workspace/Video-Depth-Anything/datasets/KITTI"  # 현재 우리 docker 안의 경로로 설정함

    print(f"데이터셋 루트 경로: {kitti_path}\n")

    # ------------------------------------------------------------------------
    # 2) 학습/검증 데이터셋 생성
    train_dataset = KITTIVideoDataset(
        root_dir=kitti_path,
        clip_len=8,
        resize_size=518,
        split="train"
    )
    val_dataset = KITTIVideoDataset(
        root_dir=kitti_path,
        clip_len=8,
        resize_size=518,
        split="val"
    )

    # ------------------------------------------------------------------------
    # 3) "비디오 폴더" 단위 개수 출력
    print("===== 데이터셋 폴더 통계 =====")
    print(f"TRAIN split: 비디오 폴더 수 = {len(train_dataset.video_infos)}")  # video_pairs → video_infos
    print(f"VAL   split: 비디오 폴더 수 = {len(val_dataset.video_infos)}\n")  # video_pairs → video_infos

    # ------------------------------------------------------------------------
    # 4) "이미지 파일" 총 개수 세기
    train_frame_count = count_total_frames(train_dataset.video_infos)  # video_pairs → video_infos
    val_frame_count   = count_total_frames(val_dataset.video_infos)    # video_pairs → video_infos
    total_frame_count = train_frame_count + val_frame_count

    print("===== 데이터셋 이미지(프레임) 통계 =====")
    print(f"TRAIN split: 이미지 파일 개수 = {train_frame_count}")
    print(f"VAL   split: 이미지 파일 개수 = {val_frame_count}")
    print(f"전체 합계    : 이미지 파일 개수 = {total_frame_count}  ← 예상 42520장인지 확인\n")

    # ------------------------------------------------------------------------
    # 5) 단일 샘플(클립) 불러오기 확인
    print("----- 단일 샘플(클립) 불러오기 테스트 (train_dataset[0]) -----")
    rgb_clip, depth_clip, mask_clip = train_dataset[0]  # 마스크도 함께 반환
    
    print(f"RGB 클립 텐서 shape  : {rgb_clip.shape}")
    print(f"Depth 클립 텐서 shape: {depth_clip.shape}")
    print(f"Mask 클립 텐서 shape: {mask_clip.shape}")  # 마스크 shape도 출력
    print(f"RGB 데이터 타입       : {rgb_clip.dtype}")
    print(f"Depth 데이터 타입     : {depth_clip.dtype}")
    print(f"RGB 값 범위 (min, max): ({rgb_clip.min():.3f}, {rgb_clip.max():.3f})")
    print(f"Depth 값 범위 (min, max): ({depth_clip.min():.3f}, {depth_clip.max():.3f})")
    print(f"RGB 평균값            : {rgb_clip.mean():.3f}")
    print(f"Depth 평균값          : {depth_clip.mean():.3f}\n")

    # ------------------------------------------------------------------------
    # 6) DataLoader로부터 배치 단위로 불러오기 테스트
    batch_size = 1
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )

    print(f"----- train_loader 첫 배치 (batch_size={batch_size}) 가져오기 -----")
    rgb_batch, depth_batch, mask_batch = next(iter(train_loader))  # 마스크도 함께 가져옴
    print(f"train_batch RGB shape: {rgb_batch.shape}")
    print(f"train_batch Depth shape: {depth_batch.shape}")
    print(f"train_batch Mask shape: {mask_batch.shape}\n")  # 마스크 shape도 출력
    
    print(f"----- val_loader 첫 배치 (batch_size={batch_size}) 가져오기 -----")
    rgb_val_batch, depth_val_batch, mask_val_batch, extr_params, intr_params = next(iter(val_loader))
    print(f"val_batch RGB shape  : {rgb_val_batch.shape}")
    print(f"val_batch Depth shape: {depth_val_batch.shape}")
    print(f"val_batch Mask shape : {mask_val_batch.shape}")

    # extr_params, intr_params는 리스트이므로 길이를 출력하거나 내부 형태를 확인합니다.
    print(f"val_batch Extrinsic params: list of length {len(extr_params)}")
    print(f"  → 첫 번째 시퀀스의 프레임 수: {len(extr_params[0])}  (각 프레임별 4×4 행렬 또는 None)")
    print(f"val_batch Intrinsic params: list of length {len(intr_params)}")
    print(f"  → 첫 번째 시퀀스의 프레임 수: {len(intr_params[0])}  (각 프레임별 [fx,fy,cx,cy] 또는 None)\n")

    # ------------------------------------------------------------------------
    # 7) 시각화: train 첫 배치의 첫 번째 클립에서 첫 번째 프레임 (RGB와 Depth 모두)
    os.makedirs("test_output", exist_ok=True)
    
    # RGB 프레임 시각화
    rgb_sample_frame = rgb_batch[0, 0].cpu().numpy().transpose(1, 2, 0)
    # 정규화된 이미지를 시각화를 위해 다시 원래 범위로 변환
    rgb_mean = np.array(train_dataset.rgb_mean)
    rgb_std = np.array(train_dataset.rgb_std)
    rgb_sample_frame = rgb_sample_frame * rgb_std[np.newaxis, np.newaxis, :] + rgb_mean[np.newaxis, np.newaxis, :]
    rgb_sample_frame = np.clip(rgb_sample_frame, 0, 1)
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_sample_frame)
    plt.title("RGB: Batch[0], Clip[0], Frame[0]")
    plt.axis("off")
    
    # Depth 프레임 시각화
    depth_sample_frame = depth_batch[0, 0].cpu().numpy().transpose(1, 2, 0)
    plt.subplot(1, 2, 2)
    plt.imshow(depth_sample_frame)
    plt.title("Depth: Batch[0], Clip[0], Frame[0]")
    plt.axis("off")
    
    plt.tight_layout()
    plt.savefig("test_output/sample_frames_rgb_depth.png", bbox_inches="tight")
    plt.close()
    print("RGB와 Depth 샘플 프레임 이미지 저장됨: test_output/sample_frames_rgb_depth.png")

     # ------------------------------------------------------------------------
    # 8) 시각화: 클립 내 모든 프레임을 한 행에 (RGB, Depth) 쌍으로 보여주기
    print("\n----- 클립 내 모든 프레임 시각화 (각 행: RGB + Depth) -----")
    num_frames = rgb_batch.shape[1]  # 실제 클립 길이 (예: 8)
    plt.figure(figsize=(6, 3 * num_frames))  # (너비=6인치, 높이=3*num_frames인치)

    for i in range(num_frames):
        # RGB
        plt.subplot(num_frames, 2, i * 2 + 1)
        rgb_frame = rgb_batch[0, i].cpu().numpy().transpose(1, 2, 0)
        rgb_frame = rgb_frame * rgb_std[np.newaxis, np.newaxis, :] + rgb_mean[np.newaxis, np.newaxis, :]
        rgb_frame = np.clip(rgb_frame, 0, 1)
        plt.imshow(rgb_frame)
        plt.title(f"RGB Frame {i}")
        plt.axis("off")

        # Depth
        plt.subplot(num_frames, 2, i * 2 + 2)
        depth_frame = depth_batch[0, i].cpu().numpy().transpose(1, 2, 0)
        plt.imshow(depth_frame)
        plt.title(f"Depth Frame {i}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("test_output/clip_frames_rgb_depth.png", bbox_inches="tight")
    plt.close()
    print("RGB와 Depth 클립 프레임 이미지 저장됨: test_output/clip_frames_rgb_depth.png")

    print("\n테스트 완료!")


def metric_val(infs, disparity_gts, gts, valid_mask):

    """
    least square 때문에, i & i+1 계산보다는 클립 하나를 통째로 계산하는 것이 올바름 
    infs,gts : [clip_len, ~] 꼴을 기대, inf는 

    지금 문제 : gt가 여기서는 진짜 gt를 기대하고있음 clipping 하기 전 ..
    """

    ### 1. preprocessing
    
    infs = torch.clamp(infs, min=1e-3)
    pred_disp_masked = infs[valid_mask].view(-1, 1).double()
    disparity_gts_disp_masked = disparity_gts[valid_mask].view(-1, 1).double()

    ### 2. least square
    
    _ones = torch.ones_like(pred_disp_masked)
    A = torch.cat([pred_disp_masked, _ones], dim=-1) 
    X = torch.linalg.lstsq(A, disparity_gts_disp_masked).solution  
    scale = X[0].item()
    shift = X[1].item()
    aligned_pred = scale * infs + shift
    aligned_pred = torch.clamp(aligned_pred, min=1e-3)
    
    print("aligned_pred : ",aligned_pred[0][0])
    print("disparity_gts_disp_masked" ,disparity_gts_disp_masked[0][0])

    ### 3. recovery
    
    depth = torch.zeros_like(aligned_pred)
    depth = 1.0 / aligned_pred
    
    gt_depth = gts
    pred_depth = torch.clamp(depth, min=1e-3, max=MAX_DEPTH)
    
    #print("scaled_pred_depth : ",pred_depth[0][0])
    #print("gt_depth : ",gt_depth[0][0])

    ### 4. validity
    n = valid_mask.sum((-1, -2))
    valid_frame = (n > 0)
    pred_depth = pred_depth[valid_frame]
    gt_depth = gt_depth[valid_frame]
    valid_mask = valid_mask[valid_frame]
    
    #print("valid_mask : ",valid_mask[0])

    absrel = abs_relative_difference(pred_depth, gt_depth, valid_mask)
    delta1 = delta1_acc(pred_depth, gt_depth, valid_mask)
    #tae = eval_tae(pred_depth, poses, Ks, valid_mask)

    return absrel,delta1


def eval_tae(depth1, depth2, pose1, pose2, K1, K2, mask1, mask2):

    device = depth1.device

    # 1) pose1 → pose2 로 변환
    T_1 = pose1.to(device)
    T_2 = pose2.to(device)
    T_2_inv = torch.linalg.inv(T_2)
    T_2_1 = T_2_inv @ T_1       # 4×4

    R_2_1 = T_2_1[:3, :3]       # [3, 3]
    t_2_1 = T_2_1[:3,  3]       # [3]
    
    if mask1 is None:
        mask1 = torch.ones_like(depth1, dtype=torch.bool, device=device)
    if mask2 is None:
        mask2 = torch.ones_like(depth2, dtype=torch.bool, device=device)

    error1 = tae_torch(
        depth1,      
        depth2,       
        R_2_1,       
        t_2_1,      
        K1.to(device),
        mask2        
    )

    T_1_2 = torch.linalg.inv(T_2_1)
    R_1_2 = T_1_2[:3, :3]
    t_1_2 = T_1_2[:3,   3]

    error2 = tae_torch(
        depth2,  
        depth1,       
        R_1_2,       
        t_1_2,
        K2.to(device),
        mask1         
    )

    result = 0.5 * (error1 + error2)
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
    num_epochs = hyper_params["epochs"]
    patient = hyper_params["patient"]
    batch_size = hyper_params["batch_size"]
    conv_out_channel = hyper_params["conv_out_channel"]
    conv = hyper_params["conv"]


    ### 2. Load data

    kitti_path = "/workspace/Video-Depth-Anything/datasets/KITTI"

    # 2) 학습/검증 데이터셋 생성
    train_dataset = KITTIVideoDataset(
        root_dir=kitti_path,
        clip_len=2,
        resize_size=518,
        split="train"
    )
    val_dataset = KITTIVideoDataset(
        root_dir=kitti_path,
        clip_len=2,
        resize_size=518,
        split="val"
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)


    ### 3. Model and additional stuffs,...

    model = VideoDepthAnything(out_channel=conv_out_channel,conv=conv).to(device)
    """
    out_channel : result of diff-conv
    conv : usage. False to use raw RGB diff
    """
    
    # freeze -> pretrain은 DINO밖에 없어서 이렇게 가능 
    for param in model.pretrained.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters:", total_params)

    loss_tgm = Loss_tgm()
    loss_ssi = Loss_ssi()

    wandb.watch(model, log="all")

    best_val_loss = float('inf')
    best_epoch = 0
    trial = 0

    scaler = GradScaler()

    ### 4. train
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (x, y, masks) in tqdm(enumerate(train_loader)):
            x, y, masks = x.to(device), y.to(device), masks.to(device)
            
            
            if epoch == 0 and batch_idx == 0:
                import matplotlib.pyplot as plt
                # CPU로 이동시키고 NumPy 변환
                x_np = x[0].cpu().numpy()      # shape: [T, 3, H, W]
                y_np = y[0, 0, 0].cpu().numpy()   # shape: [H, W] (첫 번째 프레임의 disparity)

                # (1) 입력 RGB 시각화: 첫 번째 클립 첫 번째 프레임
                rgb_frame = x_np[0].transpose(1, 2, 0)  # [H, W, 3]
                # 정규화된 상태일 경우 원래 범위(0~1)로 클램핑
                rgb_vis = np.clip(rgb_frame, 0, 1)

                plt.figure(figsize=(6, 6))
                plt.imshow(rgb_vis)
                plt.title("Debug: Input X (Epoch 0, Batch 0) – First Clip, First Frame (RGB)")
                plt.axis("off")
                plt.savefig("debug_input_rgb.png", bbox_inches="tight")
                plt.close()

                # (2) GT Disparity 시각화: 첫 번째 클립 첫 번째 프레임
                plt.figure(figsize=(6, 6))
                plt.imshow(y_np, cmap="inferno")
                plt.colorbar(label="Disparity")
                plt.title("Debug: Ground Truth Y (Epoch 0, Batch 0) – First Clip, First Frame Disparity")
                plt.axis("off")
                plt.savefig("debug_gt_disparity.png", bbox_inches="tight")
                plt.close()

            #print("x:", x)  # [B, T, C, H, W]
            #print("y:", y)  # [B, T, 1, H, W]
            #print("masks:", masks)  # [B, T, 1, H, W]
            
            optimizer.zero_grad()
            
            with autocast():
                pred = model(x)  # pred.shape == [B, T, H, W]
                # masks.shape == [B, T, 1, H, W]
                
                #print("pred: ", pred)
                print("pred.sum(): ", pred.sum())
                if pred.sum()==0:
                    print("pred_sum = 0, see GT : ",y[0][0])

                y = y[:, :, 0, :, :]  # now y.shape == [B, T, H, W]

                # 마스크 채널 축 제거
                masks_squeezed = masks.squeeze(2)  # [B, T, H, W]

                # 유효 픽셀에만 곱하기
                pred_masked = pred * masks_squeezed
                y_masked    = y    * masks_squeezed

                loss_tgm_val = loss_tgm(pred_masked, y_masked, masks_squeezed)
                loss_ssi_val = loss_ssi(pred_masked, y_masked, masks_squeezed)
                loss = ratio_tgm * loss_tgm_val + ratio_ssi * loss_ssi_val
                
                # 또는 스케일링 방식 사용: 유효한 픽셀 수로 정규화
                # valid_pixel_ratio = masks.sum() / (masks.shape[0] * masks.shape[1] * masks.shape[2] * masks.shape[3] * masks.shape[4])
                # loss = (ratio_tgm * loss_tgm_val + ratio_ssi * loss_ssi_val) / (valid_pixel_ratio + 1e-8)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            
            if batch_idx == 0:
                break
              

            if batch_idx % 5 == 0:
                print(f"Epoch [{epoch}], Batch [{batch_idx}], Loss: {loss.item():.4f}")

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
            for batch_idx, (x, y, masks, true_depth, extrinsics, intrinsics) in tqdm(enumerate(val_loader)):
                x, y, masks,true_depth = x.to(device), y.to(device), masks.to(device), true_depth.to(device)
                pred = model(x)
                
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
                
                print("true_depth:", true_depth[0][0])
                print("true_depth.shape:", true_depth.shape)
                print("disparity_gt:", y[0][0])
                print("pred_depth : ",pred[0][0])
                print("pred_sum", torch.sum(pred[0][0]))  # 전체 예측값의 합계 출력
                
                
                for b in range(B):
                    inf_clip   = pred[b]         # [clip_len, H, W]
                    disparity_gt_clip = y[b]
                    gt_clip    = true_depth[b]           
                    mask_clip  = masks[b]      
                    poses_clip = poses[b]      
                    Ks_clip    = Ks[b]          

                    absrel, delta1 = metric_val(
                        inf_clip, disparity_gt_clip, gt_clip, mask_clip
                    )
                    
                    if(b+1 < B):
                        tae = eval_tae(pred[b],pred[b+1],poses[b] , poses[b+1], Ks[b], Ks[b+1], masks[b], masks[b+1])
                        total_tae += tae
                    
                    total_absrel += absrel
                    total_delta1 += delta1
                    cnt_clip += 1
                    
                # 검증 시에도 동일한 마스킹 적용
                masks_expanded = masks.expand_as(pred)
                pred_masked = pred * masks_expanded
                y_masked = y * masks_expanded
                masks_squeezed = masks.squeeze(2)
                
                # 손실 계산
                loss_tgm_val = loss_tgm(pred_masked, y_masked, masks_squeezed)
                loss_ssi_val = loss_ssi(pred_masked, y_masked, masks_squeezed)
                loss = ratio_tgm * loss_tgm_val + ratio_ssi * loss_ssi_val
                val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
        
            avg_absrel = total_absrel / cnt_clip
            avg_delta1 = total_delta1 / cnt_clip
            avg_tae = total_tae / (cnt_clip-1)

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
        """
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(), 
                'scaler_state_dict': scaler.state_dict()       
            }, 'best_checkpoint.pth')
            print(f"Best checkpoint saved at epoch {epoch} with validation loss {avg_val_loss:.4f}")
            trial = 0
        else:
            trial += 1

        if trial >= patient:
            print("Early stopping triggered.")
            break
        """

    print(f"Training finished. Best checkpoint was from epoch {best_epoch} with validation loss {best_val_loss:.4f}.")
    run.finish()

if __name__ == "__main__":
    #test_vkitti_dataloader_fullcount()
    train()


