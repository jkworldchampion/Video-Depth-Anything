#! /usr/bin/python3
import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from data import KITTIVideoDataset  # 클래스 이름 확인
import yaml
import wandb

from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.loss import Loss_ssi, Loss_tgm
from torch.cuda.amp import autocast, GradScaler
from video_depth_anything.video_depth import VideoDepthAnything

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
        clip_len=12,
        resize_size=518,
        split="train"
    )
    val_dataset = KITTIVideoDataset(
        root_dir=kitti_path,
        clip_len=12,
        resize_size=518,
        split="val"
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
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
            optimizer.zero_grad()
            
            with autocast(dtype=torch.float16):
                pred = model(x)  # pred.shape == [B, T, H, W]
                # masks.shape == [B, T, 1, H, W]

                y = y[:, :, 0, :, :]  # now y.shape == [B, T, H, W]

                # ▶ 마스크 채널 축 제거
                masks_squeezed = masks.squeeze(2)  # [B, T, H, W]

                # ▶ 유효 픽셀에만 곱하기
                pred_masked = pred * masks_squeezed
                y_masked    = y    * masks_squeezed

                loss_tgm_val = loss_tgm(pred_masked, y_masked)
                loss_ssi_val = loss_ssi(pred_masked, y_masked)
                loss = ratio_tgm * loss_tgm_val + ratio_ssi * loss_ssi_val
                
                # 또는 스케일링 방식 사용: 유효한 픽셀 수로 정규화
                # valid_pixel_ratio = masks.sum() / (masks.shape[0] * masks.shape[1] * masks.shape[2] * masks.shape[3] * masks.shape[4])
                # loss = (ratio_tgm * loss_tgm_val + ratio_ssi * loss_ssi_val) / (valid_pixel_ratio + 1e-8)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

            if batch_idx % 5 == 0:
                print(f"Epoch [{epoch}], Batch [{batch_idx}], Loss: {loss.item():.4f}")

        avg_train_loss = epoch_loss / len(train_loader)
        scheduler.step()

        print(f"Epoch [{epoch}/{num_epochs}] Train Loss: {epoch_loss:.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (x, y, masks, _, _) in tqdm(enumerate(val_loader)):
                x, y, masks = x.to(device), y.to(device), masks.to(device)
                pred = model(x)

                # y: [B, T, 3, H, W] → 단일 채널로
                y = y[:, :, 0, :, :]  # [B, T, H, W]
                masks_squeezed = masks.squeeze(2)  # [B, T, H, W]

                pred_masked = pred * masks_squeezed
                y_masked    = y    * masks_squeezed

                loss_tgm_val = loss_tgm(pred_masked, y_masked)
                loss_ssi_val = loss_ssi(pred_masked, y_masked)
                loss = ratio_tgm * loss_tgm_val + ratio_ssi * loss_ssi_val
                val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch}/{num_epochs}] Validation Loss: {avg_val_loss:.4f}")

        wandb.log({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
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
                'scaler_state_dict': scaler.state_dict()       
            }, 'best_checkpoint.pth')
            print(f"Best checkpoint saved at epoch {epoch} with validation loss {avg_val_loss:.4f}")
            trial = 0
        else:
            trial += 1

        if trial >= patient:
            print("Early stopping triggered.")
            break

    print(f"Training finished. Best checkpoint was from epoch {best_epoch} with validation loss {best_val_loss:.4f}.")
    run.finish()

if __name__ == "__main__":
    # test_vkitti_dataloader_fullcount()
    train()


