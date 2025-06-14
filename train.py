#! /usr/bin/python3
import os
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml
import wandb
import gc
import time

from torch.utils.data import DataLoader 
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from utils.loss_MiDas import Loss_ssi, LossTGMVector 
from data.VKITTI import KITTIVideoDataset
from data.Google_Landmark import GoogleLandmarksDataset, CombinedDataset
from video_depth_anything.video_depth import VideoDepthAnything
from benchmark.eval.metric import *
from benchmark.eval.eval_tae import tae_torch
from benchmark.eval.eval import depth2disparity
from utils.util import *

from PIL import Image

matplotlib.use('Agg')

MAX_DEPTH=80.0

# 초기 설정
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MEAN = torch.tensor((0.485,0.456,0.406), device=DEVICE).view(3,1,1)
STD  = torch.tensor((0.229,0.224,0.225), device=DEVICE).view(3,1,1)
# KITTI 용
MIN_DISP = 1.0/80.0
MAX_DISP = 1.0/0.001


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
    seed = hyper_params["seed"]
    threshold = hyper_params["threshold"]


    ### 2. Load data
    kitti_path = "/home/icons/workspace/SungChan/Video-Depth-Anything/datasets/KITTI"
    google_path = "/home/icons/workspace/SungChan/Video-Depth-Anything/datasets/google_landmarks"

    # 2) 학습/검증 데이터셋 생성
    # -> 원래 쓰던 dataloader에서 seed 추가해서 재현 가능하게 했고, kitti의 경우 disparity가 아닌 원래 값 전달하는 식으로 변경
    kitti_train = KITTIVideoDataset(
        root_dir=kitti_path,
        clip_len=CLIP_LEN,
        resize_size=518,
        seed=seed,
        split="train"
    )
    
    kitti_val = KITTIVideoDataset(
        root_dir=kitti_path,
        clip_len=CLIP_LEN,
        resize_size=518,
        seed=seed,
        split="val"
    )

    train_dataset = CombinedDataset(
        kitti_train,
        google_image_root=google_path + "/images",
        google_depth_root=google_path + "/depth",
        output_size=518,
        seed=seed
    )
    
    val_dataset = CombinedDataset(
        kitti_val,
        google_image_root=google_path + "/images",
        google_depth_root=google_path + "/depth",
        output_size=518,
        seed=seed
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)

    ### 3. Model and additional stuffs,...
    model = VideoDepthAnything(num_frames=CLIP_LEN,out_channel=conv_out_channel,conv=conv).to(device)
    """
    out_channel : result of diff-conv
    conv : usage. False to use raw RGB diff
    """
    
    # freeze -> pretrain은 DINO밖에 없어서 이렇게 가능 
    for param in model.pretrained.parameters():
        param.requires_grad = False
    
    # multi GPUs setting
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters:", total_params)

    loss_tgm = LossTGMVector(static_th=threshold)
    loss_ssi = Loss_ssi()

    wandb.watch(model, log="all")

    ### 4. 체크포인트 로딩(이어 학습)
    checkpoint_path = "latest_checkpoint_tgm_good.pth"
    start_epoch     = 0
    best_val_loss   = float('inf')
    best_epoch      = 0
    trial           = 0

    scaler = GradScaler()
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path} ...")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"]) # mixed training
        start_epoch   = ckpt["epoch"]
        best_val_loss = ckpt["best_val_loss"]
        best_epoch    = ckpt["best_epoch"]
        trial         = ckpt["trial"]
        print(f"Resuming from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}, trial={trial}")

    ### 4. train
    start_time = time.time()
    for epoch in tqdm(range(start_epoch, num_epochs), desc="Epoch", leave=False):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        
        # 재현성을 위한 dataloader 설정
        train_dataset.set_epoch(epoch)
        val_dataset.set_epoch(epoch)

        for batch_idx, (x, y, x_image, y_image, mask_image) in tqdm(enumerate(train_loader)):
            video_masks = get_mask(y,min_depth=0.001,max_depth=80.0)
            x, y = x.to(device), y.to(device)
            video_masks = video_masks.to(device)
            
            # point. image는 B랑 T랑 바꿔줘야 계산할때 연속하지 않다고 판단할 수 있음
            x_image = x_image.permute(1,0,2,3,4).to(device)
            y_image = y_image.permute(1,0,2,3,4).to(device)
            mask_image = mask_image.permute(1,0,2,3,4).to(device).bool()

            optimizer.zero_grad()
            with autocast():
                pred = model(x)  # pred.shape == [B, T, H, W]
                pred_image = model(x_image)
                
                print("video : pred.mean():", pred.mean().item())
                print("image : pred.mean():", pred_image.mean().item())
                
                # video ssi_loss
                disp_normed = norm_ssi(y,video_masks)
                video_masks_squeezed = video_masks.squeeze(2)
                loss_ssi_val = loss_ssi(pred, disp_normed, video_masks_squeezed)
                
                # tgm loss 계산
                loss_tgm_val = loss_tgm(pred, y, video_masks)
                
                # image ssi_loss
                img_disp_normed = norm_ssi(y_image,mask_image)
                mask_image_squeezed = mask_image.squeeze(2)
                loss_ssi_val_image = loss_ssi(pred_image, img_disp_normed, mask_image_squeezed)
                
                loss = ratio_tgm * loss_tgm_val + ratio_ssi * loss_ssi_val + ratio_ssi_image * loss_ssi_val_image
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()

            if batch_idx % 5 == 0:
                print(f"Epoch [{epoch}], Batch [{batch_idx}], Loss: {loss.item():.4f}")

        avg_train_loss = epoch_loss / len(train_loader)
        scheduler.step()

        print(f"Epoch [{epoch}/{num_epochs}] Train Loss: {epoch_loss:.4f}")
        
        wandb.log({"train_loss": avg_train_loss, "epoch": epoch})
        
        # === validation loop ===
        model.eval()
        val_loss = 0.0
        total_absrel = 0.0
        total_delta1 = 0.0
        total_tae = 0.0
        cnt_clip = 0
        
        with torch.no_grad():
            for batch_idx, (x, y, extrinsics, intrinsics) in tqdm(enumerate(val_loader)):
                # 1) move to device
                x, y = x.to(device), y.to(device)
                extrinsics, intrinsics = extrinsics.to(device), intrinsics.to(device)

                # 2) model inference + basic losses
                pred = model(x)                                        # [B, T, H, W]
                masks = get_mask(y, min_depth=0.001, max_depth=80.0)   # [B, T, 1, H, W]
                masks = masks.to(device).bool()
                disp_normed   = norm_ssi(y, masks)
                ssi_loss_val  = loss_ssi(pred, disp_normed, masks.squeeze(2))
                tgm_loss_val  = loss_tgm(pred, y, masks)
                val_loss     += ratio_ssi * ssi_loss_val + ratio_tgm * tgm_loss_val

                print("pred.mean():", pred.mean().item())

                # 3) prepare for scale & shift
                B, T, H, W = pred.shape
                MIN_DISP = 1.0 / 80.0      # depth=80m → disp=0.0125
                MAX_DISP = 1.0 / 0.001     # depth=0.001m → disp=1000.0

                raw_disp = pred.clamp(min=1e-6)                # [B, T, H, W]
                gt_disp  = (1.0 / y.clamp(min=1e-6)).squeeze(2) # [B, T, H, W]
                m_flat   = masks.squeeze(2).view(B, -1).float()# [B, P]
                p_flat   = raw_disp.view(B, -1)               # [B, P]
                g_flat   = gt_disp .view(B, -1)               # [B, P]

                # 4) build A, b for least-squares: A @ [a; b] ≈ b_vec
                A     = torch.stack([p_flat, torch.ones_like(p_flat, device=device)], dim=-1)  # [B,P,2]
                A     = A * m_flat.unsqueeze(-1)                                             # mask out invalid
                b_vec = g_flat.unsqueeze(-1) * m_flat.unsqueeze(-1)                          # [B,P,1]

                # 5) batched least-squares
                X = torch.linalg.lstsq(A, b_vec).solution  # [B,2,1]
                a = X[:,0,0].view(B,1,1,1)                 # [B,1,1,1]
                b = X[:,1,0].view(B,1,1,1)                 # [B,1,1,1]

                aligned_disp = (raw_disp * a + b).clamp(min=MIN_DISP, max=MAX_DISP)  # [B,T,H,W]

                # 4) 첫 배치에만 프레임 저장
                if batch_idx == 0:
                    save_dir = f"outputs/frames/epoch_{epoch}_batch_{batch_idx}"
                    os.makedirs(save_dir, exist_ok=True)
                    wb_images = []  # W&B 에 보낼 이미지 리스트
                    for t in range(T):
                        # a) RGB
                        rgb_norm = x[0, t]  # [3,H,W]
                        rgb_unc  = (rgb_norm * STD + MEAN).clamp(0,1)
                        rgb_np   = (rgb_unc.cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
                        Image.fromarray(rgb_np).save(os.path.join(save_dir, f"rgb_{t:02d}.png"))

                        # b) GT Disparity 저장 (depth→disparity → clamp[0,1] → 0–255)
                        depth_frame = y[0, t].squeeze(0).clamp(min=1e-6)        # [H,W]
                        disp_frame  = (1.0 / depth_frame).clamp(0,1)            # [H,W]
                        disp_np     = (disp_frame.cpu().numpy() * 255).astype(np.uint8)
                        disp_rgb_np = np.stack([disp_np]*3, axis=-1)
                        Image.fromarray(disp_rgb_np).save(os.path.join(save_dir, f"gt_{t:02d}.png"))

                        # c) Mask 저장
                        mask_frame = masks[0, t].squeeze(0).cpu().numpy().astype(np.uint8) * 255
                        Image.fromarray(mask_frame).save(os.path.join(save_dir, f"mask_{t:02d}.png"))

                        # d) Predicted Disparity 저장 (aligned_disp already disparity)
                        pred_frame = aligned_disp[0, t].cpu().numpy()           # [H,W]
                        pred_clamped = np.clip(pred_frame, 0.0, 1.0)            # [H,W]
                        pred_uint8   = (pred_clamped * 255).astype(np.uint8)
                        pred_rgb_np  = np.stack([pred_uint8]*3, axis=-1)
                        Image.fromarray(pred_rgb_np).save(os.path.join(save_dir, f"pred_{t:02d}.png"))
                        
                        # e) pred-disparity wandb에 저장
                        wb_images.append(wandb.Image(os.path.join(save_dir, f"pred_{t:02d}.png"), caption=f"pred_epoch{epoch}_frame{t:02d}"))

                    print(f"→ saved validation frames to '{save_dir}'")

                # 5) metric 평가 (모든 배치에 대해)
                for b in range(B):
                    inf_clip  = pred[b]              # [T,H,W]
                    gt_clip   = y[b].squeeze(1)      # [T,H,W]
                    mask_clip = masks[b].squeeze(1)  # [T,H,W]
                    pose      = extrinsics[b]
                    Kmat      = intrinsics[b]
                    absr, d1, tae = metric_val(inf_clip, gt_clip, pose, Kmat)
                    total_absrel  += absr
                    total_delta1  += d1
                    total_tae     += tae
                    cnt_clip     += 1

            # 최종 통계
            avg_val_loss = val_loss / len(val_loader)
            avg_absrel   = total_absrel / cnt_clip
            avg_delta1   = total_delta1 / cnt_clip
            avg_tae      = total_tae / cnt_clip

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
            "epoch": epoch,
            "pred_disparity": wb_images,
        }, step=epoch)
        
        ### best 체크포인트 저장
        is_best = False
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch+1
            is_best = True
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),  # mixed training
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'trial': trial,
            }, 'latest_checkpoint_tgm_good.pth')
            print(f"Best checkpoint saved at epoch {epoch+1} with validation loss {avg_val_loss:.4f}")
            trial = 0
        else:
            trial += 1

        ### latest(이어학습용) 체크포인트 저장
        torch.save({
            'epoch': epoch+1,  # 다음에 이어 학습할 때 시작할 epoch
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(), # mixed training
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'trial': trial,
        }, checkpoint_path)

    total_time = time.time() - start_time
    print(f"Total training time: {total_time/3600:.2f}h")

    print(f"Training finished. Best checkpoint was from epoch {best_epoch} with validation loss {best_val_loss:.4f}.")
    run.finish()

if __name__ == "__main__":
    train()
