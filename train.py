import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from data.KITTI_dataloader import KITTIVideoDataset
from torch.utils.data import DataLoader

def test_kitti_dataloader():
    # 데이터셋 경로 (필요시 수정)
    kitti_path = "/workspace/Video-Depth-Anything/datasets/KITTI"
    
    # 데이터셋 인스턴스 생성
    print(f"데이터셋 경로: {kitti_path}")
    kitti_dataset = KITTIVideoDataset(
        root_dir=kitti_path,
        clip_len=32,
        resize_size=518,
        use_rgb=False  # RGB 이미지 사용
    )
    
    # 데이터셋 기본 정보 출력
    print(f"데이터셋 크기: {len(kitti_dataset)} 비디오 클립")
    
    # 단일 샘플 불러오기
    print("단일 샘플 불러오는 중...")
    sample_clip = kitti_dataset[0]
    print(f"샘플 클립 shape: {sample_clip.shape}")
    print(f"데이터 타입: {sample_clip.dtype}")
    print(f"최소값: {sample_clip.min()}")
    print(f"최대값: {sample_clip.max()}")
    print(f"평균값: {sample_clip.mean()}")
    
    # DataLoader로 배치 불러오기 테스트
    batch_size = 2  # 메모리 사용량을 고려하여 작게 설정
    print(f"\n배치 사이즈 {batch_size}로 DataLoader 테스트 중...")
    loader = DataLoader(
        kitti_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    # 첫 배치 불러오기
    batch = next(iter(loader))
    print(f"배치 shape: {batch.shape}")
    
    # 샘플 이미지 시각화 (첫 번째 배치의 첫 번째 클립, 첫 번째 프레임)
    sample_frame = batch[0, 0].numpy().transpose(1, 2, 0)  # [H, W, C] 형태로 변환
    plt.figure(figsize=(10, 10))
    plt.imshow(sample_frame)
    plt.title("Sample Frame")
    plt.axis('off')
    
    # 저장 폴더 생성
    os.makedirs("test_output", exist_ok=True)
    plt.savefig("test_output/sample_frame.png")
    print("샘플 프레임 이미지 저장됨: test_output/sample_frame.png")
    
    # 클립 내 일부 프레임 시각화 (4x4 그리드에 16프레임 표시)
    plt.figure(figsize=(15, 15))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        frame = batch[0, i*2].numpy().transpose(1, 2, 0)  # 2프레임마다 시각화
        plt.imshow(frame)
        plt.title(f"Frame {i*2}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("test_output/clip_frames.png")
    print("클립 프레임 그리드 이미지 저장됨: test_output/clip_frames.png")
    
    print("\n테스트 완료!")
    
if __name__ == "__main__":
    test_kitti_dataloader()