import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

def get_random_crop_params(img, output_size):
    # output_size: 정사각형 crop 크기 (예제에서는 518)
    w, h = img.size
    th, tw = output_size, output_size
    if w == tw and h == th:
        return 0, 0, th, tw
    i = random.randint(0, h - th)
    j = random.randint(0, w - tw)
    return i, j, th, tw

class KITTIVideoDataset(Dataset):
    def __init__(self, root_dir, clip_len=32, resize_size=518):
        """
        root_dir: KITTI 폴더의 경로 (예: datasets/KITTI)
                  이 폴더 내에는 vkitti_2.0.3_depth, vkitti_2.0.3_rgb 가 존재함.
                  본 데이터셋은 vkitti_2.0.3_depth 데이터를 사용합니다.
        clip_len: 샘플링할 영상 클립의 길이 (프레임 수)
        resize_size: 짧은 변을 맞출 크기 (픽셀)
        """
        self.clip_len = clip_len
        self.resize_size = resize_size
        
        self.video_dirs = []
        depth_root = os.path.join(root_dir, "vkitti_2.0.3_depth")
        # Scene로 시작하는 폴더들을 순회
        for scene in sorted(os.listdir(depth_root)):
            scene_path = os.path.join(depth_root, scene)
            if not os.path.isdir(scene_path) or not scene.startswith("Scene"):
                continue
            # 각 Scene 내 조건 폴더 순회 (예: 15-deg-left, fog, 등)
            for cond in sorted(os.listdir(scene_path)):
                cond_path = os.path.join(scene_path, cond)
                if not os.path.isdir(cond_path):
                    continue
                # frames/depth/Camera_0 및 Camera_1 폴더를 video로 취급
                for cam in ["Camera_0", "Camera_1"]:
                    cam_path = os.path.join(cond_path, "frames", "depth", cam)
                    if os.path.isdir(cam_path):
                        self.video_dirs.append(cam_path)

        if len(self.video_dirs) == 0:
            raise ValueError("유효한 비디오 디렉터리가 없습니다. 경로를 확인해주세요.")

    def __len__(self):
        return len(self.video_dirs)

    def __getitem__(self, idx):
        video_path = self.video_dirs[idx]
        frame_files = sorted(os.listdir(video_path))
        if len(frame_files) < self.clip_len:
            raise ValueError(f"영상 {video_path}는 {self.clip_len} 프레임보다 적습니다.")
        # 연속된 clip_len 프레임 샘플링을 위한 시작 인덱스 랜덤 선정
        start_idx = random.randint(0, len(frame_files) - self.clip_len)
        
        # 첫 프레임을 불러오고, resize 및 crop 좌표 결정
        first_frame = Image.open(os.path.join(video_path, frame_files[start_idx])).convert("RGB")
        resized_first = TF.resize(first_frame, self.resize_size)
        crop_i, crop_j, crop_h, crop_w = get_random_crop_params(resized_first, self.resize_size)
        
        clip = []
        for k in range(start_idx, start_idx + self.clip_len):
            img_path = os.path.join(video_path, frame_files[k])
            img = Image.open(img_path).convert("RGB")
            img = TF.resize(img, self.resize_size)
            img = TF.crop(img, crop_i, crop_j, crop_h, crop_w)
            img_tensor = TF.to_tensor(img)
            clip.append(img_tensor)
        # (clip_len, C, H, W) 텐서로 반환
        clip = torch.stack(clip)
        return clip