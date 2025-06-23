import os
import random
from glob import glob
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


def get_random_crop_params(img, output_size):
    """
    랜덤으로 정사각형 영역을 잘라내기 위한 좌표를 반환합니다.
    """
    w, h = img.size
    th, tw = output_size, output_size
    if w == tw and h == th:
        return 0, 0, th, tw
    i = random.randint(0, h - th)
    j = random.randint(0, w - tw)
    return i, j, th, tw


class GoogleLandmarksDataset(Dataset):
    """
    구글 랜드마크 이미지와 대응하는 disparity(depth) 맵을 로드,
    지정된 크기로 강제 리사이즈하고 마스크를 계산합니다.
    각 폴더별 이미지-뎁스 페어 수를 검사하여 누락된 파일이 있으면 경고.
    """
    def __init__(self, image_root, depth_root, output_size=518, transform=None):
        self.image_paths = []
        self.depth_paths = []
        self.output_size = output_size
        self.transform = transform
        missing_pairs = []

        # 지원할 depth 확장자들
        exts = ['.png', '.jpg', '.npy']

        for folder in sorted(os.listdir(image_root)):
            img_dir = os.path.join(image_root, folder)
            dep_dir = os.path.join(depth_root, folder)
            if not os.path.isdir(img_dir):
                print(f"경고: 이미지 폴더 없음: {img_dir}")
                continue
            if not os.path.isdir(dep_dir):
                print(f"경고: depth 폴더 없음: {dep_dir}")
                continue

            img_files = sorted(os.listdir(img_dir))
            for fname in img_files:
                img_path = os.path.join(img_dir, fname)
                base, _ = os.path.splitext(fname)
                # depth 파일 찾기
                found = False
                for ext in exts:
                    dep_path = os.path.join(dep_dir, base + ext)
                    if os.path.isfile(dep_path):
                        self.image_paths.append(img_path)
                        self.depth_paths.append(dep_path)
                        found = True
                        break
                if not found:
                    missing_pairs.append((img_path, os.path.join(dep_dir, base + '.*')))

        if missing_pairs:
            print("경고: 다음 페어가 누락되었거나 파일명이 불일치합니다 (최대 10개):")
            for img_path, search_pattern in missing_pairs[:10]:
                print(f"  이미지 : {img_path}\n  depth (예상 경로 패턴) : {search_pattern}\n")
            if len(missing_pairs) > 10:
                print(f"  ... 및 추가 {len(missing_pairs)-10}개")

        if len(self.image_paths) == 0:
            raise ValueError("GoogleLandmarksDataset: no image-depth pairs found.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # RGB 이미지 로드 및 강제 리사이즈
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = TF.resize(img, (self.output_size, self.output_size))
        if self.transform:
            img = self.transform(img)
        x_image = TF.to_tensor(img)           # [3, H, W]

        # disparity(depth) map 로드 및 강제 리사이즈
        dep_path = self.depth_paths[idx]
        if dep_path.endswith('.npy'):
            disp = np.load(dep_path).astype(np.float32)
            dep_img = Image.fromarray(disp)
            dep_img = TF.resize(dep_img, (self.output_size, self.output_size))
            disp = np.array(dep_img, np.float32)
        else:
            dep = Image.open(dep_path).convert("F")
            dep = TF.resize(dep, (self.output_size, self.output_size))
            disp = np.array(dep, dtype=np.float32)
        y_image = torch.from_numpy(disp).unsqueeze(0)  # [1, H, W]

        # mask 계산: disparity < 1/80 또는 > 1000
        mask_bool = np.logical_or(disp < (1.0/80.0), disp > 1000.0)
        image_mask = torch.from_numpy(mask_bool).unsqueeze(0)
        
        # [0,1] 범위로 정규화
        if y_image.max() > y_image.min():
            disparity_norm = (y_image - y_image.min()) / (y_image.max() - y_image.min() + 1e-8)
        else:
            disparity_norm = y_image

        return x_image, disparity_norm, image_mask


class CombinedDataset(Dataset):
    """
    기존 KITTIVideoDataset과 GoogleLandmarksDataset을 조합하여,
    train/val 모드에 따라 서로 다른 튜플을 반환합니다.
    """
    def __init__(self, kitti_dataset, google_image_root, google_depth_root, image_transform=None, output_size=518):
        self.kitti = kitti_dataset
        self.google = GoogleLandmarksDataset(
            google_image_root,
            google_depth_root,
            output_size=output_size,
            transform=image_transform
        )

    def __len__(self):
        return min(len(self.kitti), len(self.google))

    def __getitem__(self, idx):
        k_idx = idx % len(self.kitti)
        g_idx = idx % len(self.google)

        # Kitti 데이터
        k_items = self.kitti[k_idx]
        if self.kitti.split == "train":
            x, y, masks = k_items
        else:
            x, y, masks, true_depth, extrinsics, intrinsics = k_items

        # Google 랜드마크
        x_image, y_image, image_mask = self.google[g_idx]

        if self.kitti.split == "train":
            return x, y, masks, x_image, y_image, image_mask
        else:
            return x, y, masks, true_depth, extrinsics, intrinsics
