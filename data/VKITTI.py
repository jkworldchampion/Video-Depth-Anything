#! /usr/bin/python3
import os
import random
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

def get_random_crop_params(img, output_size):
    """
    img: PIL.Image  
    output_size: int (예: 518)  
    → 이미지 크기가 (w, h)일 때, output_size×output_size로 자를 (i,j) 좌표 반환
    """
    w, h = img.size
    th, tw = output_size, output_size
    if w == tw and h == th:
        return 0, 0, th, tw
    i = random.randint(0, h - th)
    j = random.randint(0, w - tw)
    return i, j, th, tw


class KITTIVideoDataset(Dataset):
    """
    Virtual KITTI 2 데이터셋을 “32프레임짜리 클립” 단위로 읽어오는 Dataset.
    ─────────────────────────────────────────────────────────────────────────
    root_dir   : VKITTI 루트 경로 (예: "/data/VKITTI")
                  ├─ vkitti_2.0.3_rgb/
                  │   ├─ Scene0/15-deg-left/frames/rgb/Camera_0/ … 
                  │   └─ …
                  └─ vkitti_2.0.3_depth/
                      ├─ Scene0/15-deg-left/frames/depth/Camera_0/ …
                      └─ …

    clip_len   : 클립당 프레임 개수 (여기서는 32)
    resize_size: Resize 후 짧은 변 격자가 될 픽셀 크기 (예: 518)
    use_rgb    : True→RGB 영상 폴더, False→Depth 영상 폴더
    split      : "train" 혹은 "val".  
                 - "val"일 때는 **반드시** 경로에 "Scene20"이 포함된 비디오만 사용  
                 - "train"일 때는 "Scene20"이 포함되지 않은 나머지 비디오를 사용
    """

    def __init__(self, root_dir, clip_len=32, resize_size=518, use_rgb=True, split="train"):
        super().__init__()
        assert split in ["train", "val"], "split은 'train' 또는 'val' 중 하나여야 합니다."

        self.clip_len = clip_len
        self.resize_size = resize_size
        self.use_rgb = use_rgb
        self.split = split

        # vkitti 내부 경로 결정
        data_type = "vkitti_2.0.3_rgb" if use_rgb else "vkitti_2.0.3_depth"
        self.frame_folder = "rgb" if use_rgb else "depth"

        top_folder = os.path.join(root_dir, data_type)
        if not os.path.isdir(top_folder):
            raise FileNotFoundError(f"지정된 root_dir가 올바르지 않습니다: {top_folder}")

        # 1) 모든 valid 비디오 경로 수집
        all_video_dirs = []
        for scene in sorted(os.listdir(top_folder)):
            scene_path = os.path.join(top_folder, scene)
            if not os.path.isdir(scene_path) or not scene.startswith("Scene"):
                continue

            for cond in sorted(os.listdir(scene_path)):
                cond_path = os.path.join(scene_path, cond)
                if not os.path.isdir(cond_path):
                    continue

                # “frames/{rgb|depth}/Camera_0”과 “frames/{rgb|depth}/Camera_1”를 비디오로 취급
                for cam in ["Camera_0", "Camera_1"]:
                    cam_path = os.path.join(cond_path, "frames", self.frame_folder, cam)
                    if os.path.isdir(cam_path):
                        all_video_dirs.append(cam_path)

        if len(all_video_dirs) == 0:
            raise ValueError("유효한 비디오 디렉터리가 하나도 없습니다. 경로를 확인해주세요.")

        # 2) split에 따라 “Scene20 포함 여부”로 학습/검증 분리
        if split == "train":
            # “Scene20”이 경로에 포함되지 않은 비디오들
            self.video_dirs = [d for d in all_video_dirs if "Scene20" not in d]
        else:
            # split == "val": “Scene20”이 경로에 포함된 비디오들만
            self.video_dirs = [d for d in all_video_dirs if "Scene20" in d]

        if len(self.video_dirs) == 0:
            raise ValueError(f"'{split}' 세트에 사용할 비디오가 없습니다. split={split}, use_rgb={use_rgb}")

        print(f"[{split.upper()}] 총 {len(self.video_dirs)} 개의 비디오 디렉터리 로드됨 (use_rgb={use_rgb})")
        # 예시: 첫 번째 비디오 경로 출력
        print(f"[{split.upper()}] 첫 번째 비디오 예시 경로: {self.video_dirs[0]}")

    def __len__(self):
        # “비디오 디렉터리 개수”를 length로 취급
        return len(self.video_dirs)

    def __getitem__(self, idx):
        """
        idx번째 비디오 폴더에서 랜덤으로 연속 clip_len 개의 프레임을 뽑아서 반환.
        반환: (clip_tensor), shape = [clip_len, 3, 518, 518], dtype=torch.float32
        """

        video_path = self.video_dirs[idx]
        # 해당 폴더 내에 있는 모든 프레임 파일 목록 정렬
        frame_files = sorted(os.listdir(video_path))
        num_frames = len(frame_files)
        if num_frames < self.clip_len:
            raise ValueError(f"비디오 '{video_path}'에는 {self.clip_len}프레임이 존재하지 않습니다. (실제 {num_frames}프레임)")

        # 1) 연속된 clip_len 개의 프레임 구간을 랜덤하게 선택
        start_idx = random.randint(0, num_frames - self.clip_len)

        # 2) 첫 번째 프레임만 로드 → resize → 랜덤 크롭 좌표 계산
        first_frame_path = os.path.join(video_path, frame_files[start_idx])
        if self.use_rgb:
            first_img = Image.open(first_frame_path).convert("RGB")
        else:
            first_img = self.load_depth_image(first_frame_path)

        # “짧은 변이 resize_size”가 되도록 resize
        resized_first = TF.resize(first_img, self.resize_size)
        crop_i, crop_j, crop_h, crop_w = get_random_crop_params(resized_first, self.resize_size)

        # 3) 연속 clip_len 프레임을 차례로 로드 → 동일한 resize+crop → to_tensor → 리스트에 쌓기
        clip_list = []
        for frame_idx in range(start_idx, start_idx + self.clip_len):
            img_path = os.path.join(video_path, frame_files[frame_idx])
            if self.use_rgb:
                img = Image.open(img_path).convert("RGB")
            else:
                img = self.load_depth_image(img_path)

            # resize → crop → tensor
            img_resized = TF.resize(img, self.resize_size)
            img_cropped = TF.crop(img_resized, crop_i, crop_j, crop_h, crop_w)
            img_tensor = TF.to_tensor(img_cropped)  # [3, 518, 518], float32, [0,1]
            clip_list.append(img_tensor)

        # 4) clip_list (길이 clip_len) → Tensor로 스택: [clip_len, 3, 518, 518]
        clip_tensor = torch.stack(clip_list, dim=0)
        return clip_tensor

    def load_depth_image(self, path):
        """
        Virtual KITTI 2의 depth 이미지는 16비트 PNG로 저장되어 있음.
        이 함수를 통해 올바르게 float32 depth 맵으로 읽어온 뒤,
        (시각화를 위해) 8비트 그레이스케일 RGB 이미지로 변환하여 반환.
        """
        depth_png = Image.open(path)
        depth_arr = np.array(depth_png, dtype=np.uint16)

        # (1) 16비트 값 → [0,1]로 정규화 (예: 16비트 최대값 65535 기준)
        depth_norm = depth_arr.astype(np.float32) / 65535.0

        # (2) 다시 8비트로 스케일링 → 그레이스케일
        depth_8u = (depth_norm * 255.0).clip(0, 255).astype(np.uint8)
        depth_img = Image.fromarray(depth_8u, mode="L")  # 그레이스케일
        depth_img = depth_img.convert("RGB")             # 3채널(RGB)로 변환

        return depth_img
