import os
import random
from glob import glob
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from data.VKITTI import KITTIVideoDataset


def get_random_crop_params_with_rng(img, output_size, rng):
    """
    RNG 기반으로 랜덤 크롭 좌표를 뽑습니다.
    """
    w, h = img.size
    th, tw = output_size, output_size
    if w == tw and h == th:
        return 0, 0, th, tw
    i = rng.randint(0, h - th)
    j = rng.randint(0, w - tw)
    return i, j, th, tw

class GoogleLandmarksDataset(Dataset):
    """
    - 매 __getitem__마다 RNG(seed+idx+epoch)로 랜덤 샘플 선택
    - 같은 RNG로 랜덤 크롭(원한다면)도 적용 가능
    - transform 인자를 그대로 살려두었습니다.
    """
    def __init__(self,
                 image_root,
                 depth_root,
                 output_size=518,
                 transform=None,
                 rgb_mean=(0.485,0.456,0.406),
                 rgb_std=(0.229,0.224,0.225),
                 min_disp=1.0/80.0,
                 max_disp=1000.0,
                 seed=0):
        self.image_paths = []
        self.depth_paths = []
        self.output_size = output_size
        self.transform   = transform
        self.rgb_mean    = rgb_mean
        self.rgb_std     = rgb_std
        self.min_disp    = min_disp
        self.max_disp    = max_disp

        # reproducibility 필드
        self.seed  = seed
        self.epoch = 0

        missing = []
        exts = ['.png', '.jpg', '.npy']
        for folder in sorted(os.listdir(image_root)):
            img_dir = os.path.join(image_root, folder)
            dep_dir = os.path.join(depth_root, folder)
            if not os.path.isdir(img_dir) or not os.path.isdir(dep_dir):
                continue
            for fname in sorted(os.listdir(img_dir)):
                img_path = os.path.join(img_dir, fname)
                base, _ = os.path.splitext(fname)
                found = False
                for ext in exts:
                    dp = os.path.join(dep_dir, base + ext)
                    if os.path.isfile(dp):
                        self.image_paths.append(img_path)
                        self.depth_paths.append(dp)
                        found = True
                        break
                if not found:
                    missing.append((img_path, os.path.join(dep_dir, base + '.*')))
        if missing:
            print("Warning: missing", missing[:5])
        if not self.image_paths:
            raise ValueError("GoogleLandmarksDataset: no data found")

    def set_epoch(self, epoch: int):
        """학습 루프에서 epoch마다 호출해 주세요."""
        self.epoch = epoch

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1) RNG 초기화: seed + idx + epoch
        rng = random.Random(self.seed + idx + self.epoch)

        # 2) 랜덤 샘플 인덱스
        g_idx = rng.randint(0, len(self.image_paths) - 1)

        # 3) RGB 로드 → resize → optional transform
        img = Image.open(self.image_paths[g_idx]).convert("RGB")
        img = TF.resize(img, (self.output_size, self.output_size))
        # (원한다면) 랜덤 크롭도 같은 rng로:
        # ci,cj,ch,cw = get_random_crop_params_with_rng(img, self.output_size, rng)
        # img = TF.crop(img, ci, cj, ch, cw)

        if self.transform:
            img = self.transform(img)
        x_image = TF.to_tensor(img)
        x_image = TF.normalize(x_image, mean=self.rgb_mean, std=self.rgb_std)

        # 4) Disparity/Depth 로드 → resize
        dep_path = self.depth_paths[g_idx]
        if dep_path.endswith('.npy'):
            disp = np.load(dep_path).astype(np.float32)
            dimg = Image.fromarray(disp)
        else:
            dimg = Image.open(dep_path).convert("F")
        dimg = TF.resize(dimg, (self.output_size, self.output_size))
        # if doing the same crop:
        # dimg = TF.crop(dimg, ci, cj, ch, cw)

        disp = np.array(dimg, dtype=np.float32)
        # 0–1 정규화
        disp_norm = (disp - self.min_disp) / (self.max_disp - self.min_disp)
        disp_norm = np.clip(disp_norm, 0.0, 1.0).astype(np.float32)
        y_image = torch.from_numpy(disp_norm).unsqueeze(0)

        # mask
        mask_bool = (disp >= self.min_disp) & (disp <= self.max_disp)
        image_mask = torch.from_numpy(mask_bool.astype(np.float32)).unsqueeze(0)

        return x_image, y_image, image_mask


class CombinedDataset(Dataset):
    """
    KITTIVideoDataset + GoogleLandmarksDataset 을 합친 데이터셋.
    - train 시 반환: (rgb_clip, depth_clip, x_images[4], y_images[4], masks[4])
    - val   시 반환: (rgb_clip, depth_clip, extrinsics, intrinsics, x_images[4], y_images[4], masks[4])
    """
    def __init__(self,
                 kitti_dataset: KITTIVideoDataset,
                 google_image_root: str,
                 google_depth_root: str,
                 image_transform=None,
                 output_size=518,
                 seed=0):
        self.kitti = kitti_dataset
        self.google = GoogleLandmarksDataset(
            google_image_root,
            google_depth_root,
            output_size=output_size,
            transform=image_transform,
            seed=seed  # 동일 seed 사용
        )

    def set_epoch(self, epoch: int):
        """학습 루프에서 매 epoch마다 호출"""
        self.kitti.set_epoch(epoch)
        self.google.set_epoch(epoch)

    def __len__(self):
        # google 에서 4개씩 뽑으므로 //4
        return min(len(self.kitti), len(self.google) // 4)

    def __getitem__(self, idx):
        # 1) Kitti에서 16프레임 클립 가져오기
        k_items = self.kitti[idx]
        if self.kitti.split == "train":
            rgb_clip, depth_clip = k_items
        else:
            rgb_clip, depth_clip, extrinsics, intrinsics = k_items

        # 2) Google에서 4장 묶음으로 가져오기
        g_base = idx * 4
        x_imgs, y_imgs, masks = [], [], []
        for i in range(4):
            gi = (g_base + i) % len(self.google)
            x_i, y_i, m_i = self.google[gi]
            x_imgs.append(x_i)
            y_imgs.append(y_i)
            masks.append(m_i)

        # 리스트 → 텐서 [4, C, H, W]
        x_images = torch.stack(x_imgs)
        y_images = torch.stack(y_imgs)
        masks4   = torch.stack(masks)

        # 3) 최종 반환
        if self.kitti.split == "train":
            return rgb_clip, depth_clip, x_images, y_images # image는 disparity
        else:
            return rgb_clip, depth_clip, extrinsics, intrinsics