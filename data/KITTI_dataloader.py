import os
import random
from PIL import Image
import torch
import numpy as np
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
    def __init__(self, root_dir, clip_len=32, resize_size=518, use_rgb=True):
        """
        root_dir: KITTI 폴더의 경로 (예: datasets/KITTI)
                  이 폴더 내에는 vkitti_2.0.3_depth, vkitti_2.0.3_rgb 가 존재함.
        clip_len: 샘플링할 영상 클립의 길이 (프레임 수)
        resize_size: 짧은 변을 맞출 크기 (픽셀)
        use_rgb: True이면 RGB 이미지 사용, False이면 depth 이미지 사용
        """
        self.clip_len = clip_len
        self.resize_size = resize_size
        self.use_rgb = use_rgb
        
        # 타입에 따라 루트 폴더 결정
        if use_rgb:
            data_type = "vkitti_2.0.3_rgb"
            self.frame_folder = "rgb"
        else:
            data_type = "vkitti_2.0.3_depth"
            self.frame_folder = "depth"
            
        self.video_dirs = []
        data_root = os.path.join(root_dir, data_type)
        # Scene로 시작하는 폴더들을 순회
        for scene in sorted(os.listdir(data_root)):
            scene_path = os.path.join(data_root, scene)
            if not os.path.isdir(scene_path) or not scene.startswith("Scene"):
                continue
            # 각 Scene 내 조건 폴더 순회 (예: 15-deg-left, fog, 등)
            for cond in sorted(os.listdir(scene_path)):
                cond_path = os.path.join(scene_path, cond)
                if not os.path.isdir(cond_path):
                    continue
                # frames/depth(또는 rgb)/Camera_0 및 Camera_1 폴더를 video로 취급
                for cam in ["Camera_0", "Camera_1"]:
                    cam_path = os.path.join(cond_path, "frames", self.frame_folder, cam)
                    if os.path.isdir(cam_path):
                        self.video_dirs.append(cam_path)

        if len(self.video_dirs) == 0:
            raise ValueError("유효한 비디오 디렉터리가 없습니다. 경로를 확인해주세요.")
            
        print(f"총 {len(self.video_dirs)} 개의 비디오 디렉터리 로드됨")
        print(f"첫 번째 비디오 경로: {self.video_dirs[0]}")

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
        first_img_path = os.path.join(video_path, frame_files[start_idx])
        
        if self.use_rgb:
            # RGB 이미지 로드
            first_frame = Image.open(first_img_path).convert("RGB")
        else:
            # Depth 이미지 로드 (특수 처리 필요)
            first_frame = self.load_depth_image(first_img_path)
            
        resized_first = TF.resize(first_frame, self.resize_size)
        crop_i, crop_j, crop_h, crop_w = get_random_crop_params(resized_first, self.resize_size)
        
        clip = []
        for k in range(start_idx, start_idx + self.clip_len):
            img_path = os.path.join(video_path, frame_files[k])
            
            if self.use_rgb:
                # RGB 이미지 로드
                img = Image.open(img_path).convert("RGB")
            else:
                # Depth 이미지 로드 (특수 처리 필요)
                img = self.load_depth_image(img_path)
                
            img = TF.resize(img, self.resize_size)
            img = TF.crop(img, crop_i, crop_j, crop_h, crop_w)
            img_tensor = TF.to_tensor(img)
            clip.append(img_tensor)
            
        # (clip_len, C, H, W) 텐서로 반환
        clip = torch.stack(clip)
        return clip
        
    def load_depth_image(self, path):
        """Virtual KITTI 2 depth 이미지를 올바르게 로드하는 함수"""
        # PNG 이미지를 RGB 모드로 로드
        depth_png = Image.open(path)
        
        # 특수 깊이 값 변환 (Virtual KITTI 2 포맷에 맞게)
        depth_array = np.array(depth_png, dtype=np.float32)
        
        # 논문에 명시된 방식으로 스케일링 (필요시 수정)
        # 일반적으로 깊이는 정규화 필요
        depth_array = depth_array / 65535.0  # 16비트 PNG의 경우
        
        # 시각화를 위한 의사 컬러맵 적용 (선택적)
        # 여기서는 그레이스케일로 표현
        depth_image = Image.fromarray((depth_array * 255).astype(np.uint8))
        depth_image = depth_image.convert("RGB")
        
        return depth_image