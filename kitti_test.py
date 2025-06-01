#! /usr/bin/python3
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from data.VKITTI import KITTIVideoDataset  # 클래스 이름과 경로 수정

def count_total_frames(video_pairs):
    """
    video_pairs: (rgb_path, depth_path) 튜플의 리스트
    → 각 폴더 안의 파일 개수를 모두 합산하여 반환
    """
    total = 0
    for rgb_path, _ in video_pairs:  # RGB 파일 기준으로 카운트
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
        clip_len=32,
        resize_size=518,
        split="train"
    )
    val_dataset = KITTIVideoDataset(
        root_dir=kitti_path,
        clip_len=32,
        resize_size=518,
        split="val"
    )

    # ------------------------------------------------------------------------
    # 3) "비디오 폴더" 단위 개수 출력
    print("===== 데이터셋 폴더 통계 =====")
    print(f"TRAIN split: 비디오 폴더 수 = {len(train_dataset.video_pairs)}")
    print(f"VAL   split: 비디오 폴더 수 = {len(val_dataset.video_pairs)}\n")

    # ------------------------------------------------------------------------
    # 4) "이미지 파일" 총 개수 세기
    train_frame_count = count_total_frames(train_dataset.video_pairs)
    val_frame_count   = count_total_frames(val_dataset.video_pairs)
    total_frame_count = train_frame_count + val_frame_count

    print("===== 데이터셋 이미지(프레임) 통계 =====")
    print(f"TRAIN split: 이미지 파일 개수 = {train_frame_count}")
    print(f"VAL   split: 이미지 파일 개수 = {val_frame_count}")
    print(f"전체 합계    : 이미지 파일 개수 = {total_frame_count}  ← 예상 42520장인지 확인\n")

    # ------------------------------------------------------------------------
    # 5) 단일 샘플(클립) 불러오기 확인
    print("----- 단일 샘플(클립) 불러오기 테스트 (train_dataset[0]) -----")
    rgb_clip, depth_clip = train_dataset[0]  # 이제 RGB와 Depth를 함께 반환
    
    print(f"RGB 클립 텐서 shape  : {rgb_clip.shape}")
    print(f"Depth 클립 텐서 shape: {depth_clip.shape}")
    print(f"RGB 데이터 타입       : {rgb_clip.dtype}")
    print(f"Depth 데이터 타입     : {depth_clip.dtype}")
    print(f"RGB 값 범위 (min, max): ({rgb_clip.min():.3f}, {rgb_clip.max():.3f})")
    print(f"Depth 값 범위 (min, max): ({depth_clip.min():.3f}, {depth_clip.max():.3f})")
    print(f"RGB 평균값            : {rgb_clip.mean():.3f}")
    print(f"Depth 평균값          : {depth_clip.mean():.3f}\n")

    # ------------------------------------------------------------------------
    # 6) DataLoader로부터 배치 단위로 불러오기 테스트
    batch_size = 2
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
    rgb_batch, depth_batch = next(iter(train_loader))
    print(f"train_batch RGB shape: {rgb_batch.shape}")
    print(f"train_batch Depth shape: {depth_batch.shape}\n")

    print(f"----- val_loader 첫 배치 (batch_size={batch_size}) 가져오기 -----")
    rgb_val_batch, depth_val_batch = next(iter(val_loader))
    print(f"val_batch RGB shape  : {rgb_val_batch.shape}")
    print(f"val_batch Depth shape: {depth_val_batch.shape}\n")

    # ------------------------------------------------------------------------
    # 7) 시각화: train 첫 배치의 첫 번째 클립에서 첫 번째 프레임 (RGB와 Depth 모두)
    os.makedirs("test_output", exist_ok=True)
    
    # RGB 프레임 시각화
    rgb_sample_frame = rgb_batch[0, 0].cpu().numpy().transpose(1, 2, 0)
    # 정규화된 이미지를 시각화를 위해 다시 원래 범위로 변환
    rgb_mean = np.array(train_dataset.rgb_mean)
    rgb_std = np.array(train_dataset.rgb_std)
    rgb_sample_frame = rgb_sample_frame * rgb_std[:, np.newaxis, np.newaxis] + rgb_mean[:, np.newaxis, np.newaxis]
    rgb_sample_frame = np.clip(rgb_sample_frame, 0, 1)
    
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
    # 8) 시각화: 첫 번째 클립에서 2프레임 간격으로 8개씩 추출해 4×4 그리드 (RGB와 Depth)
    print("\n----- 클립 내 일부 프레임 그리드 시각화 (4x4, 총 16프레임) -----")
    plt.figure(figsize=(16, 16))
    
    for i in range(8):
        # RGB 프레임
        plt.subplot(4, 4, i*2 + 1)
        frame_idx = i * 4  # 4프레임 간격으로 선택
        frame = rgb_batch[0, frame_idx].cpu().numpy().transpose(1, 2, 0)
        # 정규화 복원
        frame = frame * rgb_std[:, np.newaxis, np.newaxis] + rgb_mean[:, np.newaxis, np.newaxis]
        frame = np.clip(frame, 0, 1)
        plt.imshow(frame)
        plt.title(f"RGB Frame {frame_idx}")
        plt.axis("off")
        
        # Depth 프레임
        plt.subplot(4, 4, i*2 + 2)
        frame = depth_batch[0, frame_idx].cpu().numpy().transpose(1, 2, 0)
        plt.imshow(frame)
        plt.title(f"Depth Frame {frame_idx}")
        plt.axis("off")
    
    plt.tight_layout()
    plt.savefig("test_output/clip_frames_rgb_depth.png", bbox_inches="tight")
    plt.close()
    print("RGB와 Depth 클립 프레임 그리드 이미지 저장됨: test_output/clip_frames_rgb_depth.png")

    print("\n테스트 완료!")

if __name__ == "__main__":
    test_vkitti_dataloader_fullcount()
