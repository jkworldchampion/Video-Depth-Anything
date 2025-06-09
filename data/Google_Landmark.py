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
    GoogleLandmarks에서 생성된 이미지를 로드하고,
    대응하는 .npy disparity를 불러와 Mask 계산
    (Kitti와 동일한 전처리: ImageNet 정규화, 518 crop)
    """
    def __init__(self, image_root, depth_root,
                 clip_len=1, resize_size=518,
                 rgb_mean=(0.485,0.456,0.406),
                 rgb_std=(0.229,0.224,0.225),
                 min_disp=1.0/80.0, max_disp=1000.0):
        super().__init__()
        self.image_paths = sorted(glob(os.path.join(image_root, '**', '*.*'), recursive=True))
        self.depth_paths = [p.replace(image_root, depth_root).rsplit('.',1)[0] + '.npy'
                            for p in self.image_paths]
        assert len(self.image_paths)==len(self.depth_paths), "image/depth count mismatch"
        self.clip_len = clip_len
        self.resize_size = resize_size
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std
        self.min_disp = min_disp
        self.max_disp = max_disp
        # simple check
        for dp in self.depth_paths:
            if not os.path.isfile(dp):
                raise FileNotFoundError(f"Depth file not found: {dp}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 단일 프레임만 처리 (clip_len=1)
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = TF.resize(img, self.resize_size)
        crop_i, crop_j, th, tw = get_random_crop_params(img, self.resize_size)
        img = TF.crop(img, crop_i, crop_j, th, tw)
        x = TF.to_tensor(img)
        x = TF.normalize(x, mean=self.rgb_mean, std=self.rgb_std)

        # disparity load
        disp = np.load(self.depth_paths[idx]).astype(np.float32)
        # disparity는 이미 518로 저장, crop만
        disp_img = Image.fromarray(disp)
        disp_img = TF.crop(disp_img, crop_i, crop_j, th, tw)
        disp_crop = np.array(disp_img, np.float32)
        y = torch.from_numpy(disp_crop).unsqueeze(0)

        # mask: Kitti 기준과 동일하게 valid disparity 범위
        mask = (disp_crop>=self.min_disp) & (disp_crop<=self.max_disp)
        mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)

        return x.unsqueeze(0), y.unsqueeze(0), mask.unsqueeze(0)  # [1,C,H,W]→[1,1,C,H,W] 등


class CombinedDataset(Dataset):
    """
    Kitti와 GoogleLandmarks를 조합. train/val에 따라 반환형 다름
    """
    def __init__(self, kitti_dataset, google_image_root, google_depth_root):
        self.kitti  = kitti_dataset
        self.google = GoogleLandmarksDataset(
            image_root=google_image_root,
            depth_root=google_depth_root,
            clip_len=1,
            resize_size=kitti_dataset.resize_size
        )

    def __len__(self):
        return min(len(self.kitti), len(self.google))

    def __getitem__(self, idx):
        k_idx = idx % len(self.kitti)
        g_idx = idx % len(self.google)

        # Kitti
        k_items = self.kitti[k_idx]
        if self.kitti.split=='train':
            x, y, masks = k_items             # [T,3,H,W], [T,3,H,W],[T,1,H,W]
        else:
            x, y, masks, true_depth, extrinsics, intrinsics = k_items

        # Google
        xg, yg, mg = self.google[g_idx]      # [1,3,H,W], [1,1,H,W], [1,1,H,W]

        if self.kitti.split=='train':
            return x, y, masks, xg, yg, mg
        else:
            return x, y, masks, true_depth, extrinsics, intrinsics
