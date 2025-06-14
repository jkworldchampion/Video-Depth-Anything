import os
import torch
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import random

from data.VKITTI import KITTIVideoDataset, get_random_crop_params_with_rng, get_center_crop_params

def compute_stats_per_folder(dataset: KITTIVideoDataset, device='cuda'):
    for idx, info in enumerate(dataset.video_infos):
        # 통계용 변수 초기화
        rgb_min, rgb_max = float('inf'), float('-inf')
        rgb_sum, rgb_count = 0.0, 0
        depth_min, depth_max = float('inf'), float('-inf')
        depth_sum, depth_count = 0.0, 0

        # 파일 리스트 (정렬 기준은 dataset.__getitem__ 과 동일하게)
        rgb_files   = sorted([f for f in os.listdir(info['rgb_path'])   if f.lower().endswith(dataset.IMG_EXTENSIONS)])
        depth_files = sorted([f for f in os.listdir(info['depth_path']) if f.lower().endswith(dataset.IMG_EXTENSIONS)])
        N = len(rgb_files)

        # seed·epoch 준비 (train일 때)
        rng = None
        if dataset.split == "train":
            rng = random.Random(dataset.seed + idx + dataset.epoch)

        # crop 좌표 결정 (첫 프레임 기준)
        first_img = Image.open(os.path.join(info['rgb_path'], rgb_files[0])).convert("RGB")
        first_img = TF.resize(first_img, dataset.resize_size)
        if dataset.split == "train":
            ci, cj, ch, cw = get_random_crop_params_with_rng(first_img, dataset.resize_size, rng)
        else:
            ci, cj, ch, cw = get_center_crop_params(first_img, dataset.resize_size)

        # 각 프레임 순회
        for rf, df in zip(rgb_files, depth_files):
            # ----- RGB 처리 -----
            img = Image.open(os.path.join(info['rgb_path'], rf)).convert("RGB")
            img = TF.resize(img, dataset.resize_size)
            if dataset.split == "train":
                img = TF.crop(img, ci, cj, ch, cw)
            else:
                img = TF.center_crop(img, dataset.resize_size)
            t = TF.to_tensor(img)
            t = TF.normalize(t, mean=dataset.rgb_mean, std=dataset.rgb_std)

            # ----- Depth 처리 -----
            d_pil = dataset.load_depth(os.path.join(info['depth_path'], df))
            d_pil = TF.resize(d_pil, dataset.resize_size)
            if dataset.split == "train":
                d_pil = TF.crop(d_pil, ci, cj, ch, cw)
            else:
                d_pil = TF.center_crop(d_pil, dataset.resize_size)
            d = TF.to_tensor(d_pil)

            # GPU로 옮겨 연산
            t = t.to(device)
            d = d.to(device)

            # RGB 통계 업데이트
            rgb_min = min(rgb_min, float(t.min()))
            rgb_max = max(rgb_max, float(t.max()))
            rgb_sum += float(t.sum())
            rgb_count += t.numel()

            # Depth 통계 업데이트
            depth_min = min(depth_min, float(d.min()))
            depth_max = max(depth_max, float(d.max()))
            depth_sum += float(d.sum())
            depth_count += d.numel()

        # mean 계산
        rgb_mean   = rgb_sum   / rgb_count
        depth_mean = depth_sum / depth_count

        print(f"\n--- Folder {idx+1}/{len(dataset.video_infos)} ---")
        print(f"Scene: {info['scene']}, Cond: {info['condition']}, Cam: {info['camera']}")
        print(f"RGB   : min={rgb_min:.4f}, max={rgb_max:.4f}, mean={rgb_mean:.4f}")
        print(f"Depth : min={depth_min:.4f}, max={depth_max:.4f}, mean={depth_mean:.4f}")



# 사용 예시
if __name__ == "__main__":
    # train 데이터 체크
    train_ds = KITTIVideoDataset(root_dir="/home/icons/workspace/SungChan/Video-Depth-Anything/datasets/KITTI", split="train")
    compute_stats_per_folder(train_ds, device='cuda')

    # val 데이터 체크
    val_ds = KITTIVideoDataset(root_dir="/home/icons/workspace/SungChan/Video-Depth-Anything/datasets/KITTI", split="val")
    compute_stats_per_folder(val_ds, device='cuda')
