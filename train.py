#! /usr/bin/python3
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from data import VKITTIVideoDataset  # 클래스 이름 확인

def count_total_frames(video_dirs):
    """
    video_dirs: 비디오 폴더 경로들이 담긴 리스트
    → 각 폴더 안의 파일 개수를 모두 합산하여 반환
    """
    total = 0
    for vpath in video_dirs:
        # 이미지 파일만 세기 (".png" 등 확장자 기준)
        files = [f for f in os.listdir(vpath) 
                 if os.path.isfile(os.path.join(vpath, f)) 
                 and (f.lower().endswith(".png") or f.lower().endswith(".jpg"))]
        total += len(files)
    return total

def test_vkitti_dataloader_fullcount():
    # ------------------------------------------------------------------------
    # 1) 데이터셋 루트 경로 (환경에 맞게 수정)
    kitti_path = "/workspace/Video-Depth-Anything/datasets/KITTI"

    print(f"데이터셋 루트 경로: {kitti_path}\n")

    # ------------------------------------------------------------------------
    # 2) 학습/검증 데이터셋 생성
    train_dataset = VKITTIVideoDataset(
        root_dir=kitti_path,
        clip_len=32,
        resize_size=518,
        use_rgb=False,
        split="train"
    )
    val_dataset = VKITTIVideoDataset(
        root_dir=kitti_path,
        clip_len=32,
        resize_size=518,
        use_rgb=False,
        split="val"
    )

    # ------------------------------------------------------------------------
    # 3) “비디오 폴더” 단위 개수 출력
    print("===== 데이터셋 폴더 통계 =====")
    print(f"TRAIN split: 비디오 폴더 수 = {len(train_dataset.video_dirs)}")
    print(f"VAL   split: 비디오 폴더 수 = {len(val_dataset.video_dirs)}\n")

    # ------------------------------------------------------------------------
    # 4) “이미지 파일” 총 개수 세기
    train_frame_count = count_total_frames(train_dataset.video_dirs)
    val_frame_count   = count_total_frames(val_dataset.video_dirs)
    total_frame_count = train_frame_count + val_frame_count

    print("===== 데이터셋 이미지(프레임) 통계 =====")
    print(f"TRAIN split: 이미지 파일 개수 = {train_frame_count}")
    print(f"VAL   split: 이미지 파일 개수 = {val_frame_count}")
    print(f"전체 합계    : 이미지 파일 개수 = {total_frame_count}  ← 예상 42520장인지 확인\n")

    # ------------------------------------------------------------------------
    # 5) 단일 샘플(클립) 불러오기 확인
    print("----- 단일 샘플(클립) 불러오기 테스트 (train_dataset[0]) -----")
    clip = train_dataset[0]  # shape = [32, 3, 518, 518]
    print(f"클립 텐서 shape   : {clip.shape}")
    print(f"데이터 타입       : {clip.dtype}")
    print(f"값 범위 (min, max): ({clip.min():.3f}, {clip.max():.3f})")
    print(f"평균값            : {clip.mean():.3f}\n")

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
    train_batch = next(iter(train_loader))
    print(f"train_batch shape: {train_batch.shape}\n")

    print(f"----- val_loader 첫 배치 (batch_size={batch_size}) 가져오기 -----")
    val_batch = next(iter(val_loader))
    print(f"val_batch shape  : {val_batch.shape}\n")

    # ------------------------------------------------------------------------
    # 7) 시각화: train 첫 배치의 첫 번째 클립에서 첫 번째 프레임
    os.makedirs("test_output", exist_ok=True)
    sample_frame = train_batch[0, 0].cpu().numpy().transpose(1, 2, 0)
    plt.figure(figsize=(6, 6))
    plt.imshow(sample_frame)
    plt.title("Train: Batch[0], Clip[0], Frame[0]")
    plt.axis("off")
    plt.savefig("test_output/sample_frame.png", bbox_inches="tight")
    plt.close()
    print("샘플 프레임 이미지 저장됨: test_output/sample_frame.png")

    # ------------------------------------------------------------------------
    # 8) 시각화: 첫 번째 클립에서 2프레임 간격으로 16개 추출해 4×4 그리드
    print("\n----- 클립 내 일부 프레임 그리드 시각화 (4x4, 총 16프레임) -----")
    plt.figure(figsize=(12, 12))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        frame_idx = i * 2
        frame = train_batch[0, frame_idx].cpu().numpy().transpose(1, 2, 0)
        plt.imshow(frame)
        plt.title(f"Frame {frame_idx}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("test_output/clip_frames.png", bbox_inches="tight")
    plt.close()
    print("클립 프레임 그리드 이미지 저장됨: test_output/clip_frames.png")

    print("\n테스트 완료!")

if __name__ == "__main__":
    test_vkitti_dataloader_fullcount()
