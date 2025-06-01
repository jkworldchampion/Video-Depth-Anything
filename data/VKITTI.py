import os
import random
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


def get_random_crop_params(img, output_size):
    """
    논문상에서 "random center cropping"라고 표현하길래,
    랜덤이야? 아니면 center crop이야? 헷갈렸는데,
    그냥 random을 좋아해서 이렇게 했어여.
    """
    w, h = img.size
    th, tw = output_size, output_size
    if w == tw and h == th:
        return 0, 0, th, tw
    i = random.randint(0, h - th)
    j = random.randint(0, w - tw)
    return i, j, th, tw


class KITTIVideoDataset(Dataset):
    def __init__(self, root_dir, clip_len=32, resize_size=518, split="train", 
                 rgb_mean=(0.485, 0.456, 0.406), rgb_std=(0.229, 0.224, 0.225)):
        super().__init__()
        assert split in ["train", "val"]  # train, val에 따라서 다른 폴더를 주기 위해 필수 요구 설정

        self.clip_len = clip_len
        self.resize_size = resize_size
        self.split = split
        self.root_dir = root_dir

        # 정규화를 위한 평균 및 표준편차 설정
        self.rgb_mean = rgb_mean  # ImageNet 평균,, KITTI에서 따로 없길래 그냥 convention으로 감
        self.rgb_std = rgb_std    # ImageNet 표준편차

        # RGB와 Depth 경로 설정
        self.rgb_root = os.path.join(root_dir, "vkitti_2.0.3_rgb")
        self.depth_root = os.path.join(root_dir, "vkitti_2.0.3_depth")

        if not os.path.isdir(self.rgb_root) or not os.path.isdir(self.depth_root):
            raise FileNotFoundError(f"RGB 또는 Depth 폴더가 존재하지 않습니다.")

        # 비디오 경로 쌍 수집 (RGB와 Depth가 일대일 대응)
        self.video_pairs = []

        for scene in sorted(os.listdir(self.rgb_root)):
            scene_rgb_path = os.path.join(self.rgb_root, scene)
            scene_depth_path = os.path.join(self.depth_root, scene)

            if not os.path.isdir(scene_rgb_path) or not os.path.isdir(scene_depth_path):
                continue

            # Scene20은 검증용으로 분리
            if (split == "train" and "Scene20" in scene) or (split == "val" and "Scene20" not in scene):
                continue

            for cond in sorted(os.listdir(scene_rgb_path)):
                cond_rgb_path = os.path.join(scene_rgb_path, cond)
                cond_depth_path = os.path.join(scene_depth_path, cond)

                if not os.path.isdir(cond_rgb_path) or not os.path.isdir(cond_depth_path):
                    continue

                for cam in ["Camera_0", "Camera_1"]:
                    rgb_path = os.path.join(cond_rgb_path, "frames", "rgb", cam)
                    depth_path = os.path.join(cond_depth_path, "frames", "depth", cam)

                    if os.path.isdir(rgb_path) and os.path.isdir(depth_path):
                        self.video_pairs.append((rgb_path, depth_path))

        if len(self.video_pairs) == 0:
            raise ValueError(f"'{split}' 세트에 사용할 비디오 쌍이 없습니다.")

        print(f"[{split.upper()}] 총 {len(self.video_pairs)} 개의 비디오 쌍 로드됨")
        print(f"[{split.upper()}] 첫 번째 비디오 쌍 예시: RGB={self.video_pairs[0][0]}, Depth={self.video_pairs[0][1]}")

    def __len__(self):
        return len(self.video_pairs)

    def __getitem__(self, idx):
        """
        RGB 이미지와 Depth 이미지 쌍을 반환
        반환: (rgb_clip, depth_clip)
        각각 shape = [clip_len, 3, 518, 518], dtype=torch.float32
        """
        rgb_path, depth_path = self.video_pairs[idx]

        # RGB와 Depth 이미지 파일 목록 정렬
        rgb_files = sorted(os.listdir(rgb_path))
        depth_files = sorted(os.listdir(depth_path))

        # 두 폴더의 파일 수가 일치하는지 확인
        if len(rgb_files) != len(depth_files):
            raise ValueError(f"RGB와 Depth의 개수가 일치 X: {rgb_path}, {depth_path}")

        # 비디오 길이 확인
        num_frames = len(rgb_files)
        if num_frames < self.clip_len:
            raise ValueError(f"비디오에 {self.clip_len}프레임이 존재하지 않습니다. (실제 {num_frames}프레임)")

        # 연속된 clip_len 프레임 구간 랜덤 선택
        start_idx = random.randint(0, num_frames - self.clip_len)

        # 첫 번째 프레임 로드하여 crop 좌표 계산
        first_rgb_path = os.path.join(rgb_path, rgb_files[start_idx])
        first_rgb = Image.open(first_rgb_path).convert("RGB")
        resized_first = TF.resize(first_rgb, self.resize_size)
        crop_i, crop_j, crop_h, crop_w = get_random_crop_params(resized_first, self.resize_size)

        # RGB 및 Depth 클립 로드
        rgb_clip = []
        depth_clip = []

        for i in range(self.clip_len):
            frame_idx = start_idx + i

            # RGB 이미지 로드
            rgb_img_path = os.path.join(rgb_path, rgb_files[frame_idx])
            rgb_img = Image.open(rgb_img_path).convert("RGB")
            rgb_resized = TF.resize(rgb_img, self.resize_size)
            rgb_cropped = TF.crop(rgb_resized, crop_i, crop_j, crop_h, crop_w)
            
            # RGB 이미지를 텐서로 변환
            rgb_tensor = TF.to_tensor(rgb_cropped)
            
            # 정규화 적용 (평균과 표준편차로)
            rgb_tensor = TF.normalize(rgb_tensor, mean=self.rgb_mean, std=self.rgb_std)
            rgb_clip.append(rgb_tensor)

            # Depth 이미지 로드
            depth_img_path = os.path.join(depth_path, depth_files[frame_idx])
            depth_img = self.load_depth_image(depth_img_path)
            depth_resized = TF.resize(depth_img, self.resize_size)
            depth_cropped = TF.crop(depth_resized, crop_i, crop_j, crop_h, crop_w)
            depth_tensor = TF.to_tensor(depth_cropped)
            
            # Depth는 일반적으로 정규화하지 않거나 다르게 정규화할 수 있음
            # depth_tensor = TF.normalize(depth_tensor, mean=[0.5], std=[0.5])  # 선택적
            
            depth_clip.append(depth_tensor)

        # 텐서로 변환
        rgb_clip_tensor = torch.stack(rgb_clip)
        depth_clip_tensor = torch.stack(depth_clip)

        return rgb_clip_tensor, depth_clip_tensor

    def load_depth_image(self, path):
        """
        Virtual KITTI 2의 depth 이미지 로드 함수
        
        깊이 이미지 설명:
        - 16비트 PNG 파일로 인코딩됨
        - 픽셀값 1 = 카메라로부터 1cm 거리 (예: 픽셀값 100 = 100cm = 1m)
        - 최대 655.35m까지 표현 가능 (최대 픽셀값 65535)
        """
        # 설명서에 있는 OpenCV를 사용하는 대신, PIL + NumPy로 동일한 작업 수행
        depth_png = Image.open(path)
    
        # 16비트 깊이 값을 float32로 변환
        depth_arr = np.array(depth_png, dtype=np.uint16).astype(np.float32)
        
        # 간단히 최대값으로 정규화 (상대적 깊이 관계 유지)
        depth_max = np.max(depth_arr)
        if depth_max > 0:  # 0으로 나누기 방지
            depth_norm = depth_arr / depth_max
        else:
            depth_norm = depth_arr
        
        # 8비트 이미지로 변환
        depth_8u = (depth_norm * 255.0).astype(np.uint8)
        depth_img = Image.fromarray(depth_8u, mode="L")  # 그레이스케일
        depth_img = depth_img.convert("RGB")             # 3채널로 변환 -> 그림으로 확인해보려고, 넣었었음
        
        return depth_img
