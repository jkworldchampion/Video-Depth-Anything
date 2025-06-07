import os
import random
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

    
class KITTIVideoDataset(Dataset):
    def __init__(self,
                 root_dir,
                 clip_len=32,
                 resize_size=518,
                 split="train",
                 rgb_mean=(0.485, 0.456, 0.406),
                 rgb_std=(0.229, 0.224, 0.225),
                 min_depth=0.001,
                 max_depth=80.0):
        super().__init__()
        assert split in ["train", "val"], "split은 'train' 또는 'val'이어야 합니다."

        self.clip_len = clip_len
        self.resize_size = resize_size
        self.split = split
        self.root_dir = root_dir
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std
        self.min_depth = min_depth
        self.max_depth = max_depth

        # VKITTI 폴더 구조 예시
        self.rgb_root = os.path.join(root_dir, "vkitti_2.0.3_rgb")
        self.depth_root = os.path.join(root_dir, "vkitti_2.0.3_depth")
        self.textgt_root = os.path.join(root_dir, "vkitti_2.0.3_textgt")

        if not os.path.isdir(self.rgb_root) or \
           not os.path.isdir(self.depth_root) or \
           not os.path.isdir(self.textgt_root):
            raise FileNotFoundError("RGB, Depth 또는 TextGT 폴더가 존재하지 않습니다.")

        # 비디오 정보 저장할 리스트
        self.video_infos = []

        for scene in sorted(os.listdir(self.rgb_root)):
            scene_rgb_path = os.path.join(self.rgb_root, scene)
            scene_depth_path = os.path.join(self.depth_root, scene)
            scene_textgt_path = os.path.join(self.textgt_root, scene)

            if not os.path.isdir(scene_rgb_path) or \
               not os.path.isdir(scene_depth_path) or \
               not os.path.isdir(scene_textgt_path):
                continue

            # Scene20은 val, 나머지는 train
            if (split == "train" and "Scene20" in scene) or \
               (split == "val" and "Scene20" not in scene):
                continue

            for condition in sorted(os.listdir(scene_rgb_path)):
                cond_rgb_path = os.path.join(scene_rgb_path, condition)
                cond_depth_path = os.path.join(scene_depth_path, condition)
                cond_textgt_path = os.path.join(scene_textgt_path, condition)

                if not os.path.isdir(cond_rgb_path) or \
                   not os.path.isdir(cond_depth_path) or \
                   not os.path.isdir(cond_textgt_path):
                    continue

                intrinsic_file = os.path.join(cond_textgt_path, "intrinsic.txt")
                extrinsic_file = os.path.join(cond_textgt_path, "extrinsic.txt")
                if not os.path.isfile(intrinsic_file) or not os.path.isfile(extrinsic_file):
                    print(f"경고: {cond_textgt_path}에 intrinsic.txt 또는 extrinsic.txt 파일이 없습니다.")
                    continue

                for cam in ["Camera_0", "Camera_1"]:
                    cam_idx = int(cam[-1])  # "Camera_0" → 0, "Camera_1" → 1
                    rgb_path = os.path.join(cond_rgb_path, "frames", "rgb", cam)
                    depth_path = os.path.join(cond_depth_path, "frames", "depth", cam)

                    if os.path.isdir(rgb_path) and os.path.isdir(depth_path):
                        self.video_infos.append({
                            'rgb_path': rgb_path,
                            'depth_path': depth_path,
                            'intrinsic_file': intrinsic_file,
                            'extrinsic_file': extrinsic_file,
                            'scene': scene,
                            'condition': condition,
                            'camera': cam_idx
                        })

        if len(self.video_infos) == 0:
            raise ValueError(f"'{split}' 세트에 사용할 비디오 쌍이 없습니다.")

        print(f"[{split.upper()}] 총 {len(self.video_infos)} 개의 비디오 쌍 로드됨")
        print(f"[{split.upper()}] 첫 번째 비디오 쌍 예시: RGB={self.video_infos[0]['rgb_path']}, Depth={self.video_infos[0]['depth_path']}")

    def __len__(self):
        return len(self.video_infos)

    def load_depth_image_with_mask(self, path):
        """
        Depth 이미지를 로드하여 disparity 이미지(PIL)와 마스크(PIL)를 반환합니다.
        """
        depth_png = Image.open(path)
        depth_cm = np.array(depth_png, dtype=np.uint16).astype(np.float32)
        depth_m = depth_cm / 100.0  # cm → m

        valid_mask = np.logical_and((depth_m > self.min_depth), (depth_m < self.max_depth))
        disparity = np.zeros_like(depth_m)
        disparity[valid_mask] = 1.0 / depth_m[valid_mask]

        # [0,1] 범위로 정규화
        if np.max(disparity) > np.min(disparity):
            disparity_norm = (disparity - np.min(disparity)) / (np.max(disparity) - np.min(disparity) + 1e-8)
        else:
            disparity_norm = disparity

        disparity_img = Image.fromarray((disparity_norm * 255.0).astype(np.uint8), mode="L")
        # disparity_img = Image.fromarray ((disparity_norm), mode="F") 
        mask_img = Image.fromarray((valid_mask * 255).astype(np.uint8), mode="L")
        disparity_img = disparity_img.convert("RGB")  # 3채널 변환

        return disparity_img, mask_img, depth_m

    @staticmethod
    def load_camera_params(intrinsic_path, extrinsic_path):
        """
        intrinsic.txt, extrinsic.txt 파일을 읽어 두 개의 딕셔너리를 반환합니다.
        - intrinsics: {(frame, camera_id): [fx, fy, cx, cy]}
        - extrinsics: {(frame, camera_id): 4x4 행렬}
        """
        intrinsics = {}
        with open(intrinsic_path, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                frame = int(parts[0])
                camera_id = int(parts[1])
                fx = float(parts[2])
                fy = float(parts[3])
                cx = float(parts[4])
                cy = float(parts[5])
                intrinsics[(frame, camera_id)] = [fx, fy, cx, cy]

        extrinsics = {}
        with open(extrinsic_path, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                parts = line.strip().split()
                if len(parts) < 18:
                    continue
                frame = int(parts[0])
                camera_id = int(parts[1])
                matrix_vals = list(map(float, parts[2:18]))
                transform = np.array(matrix_vals).reshape((4, 4))
                extrinsics[(frame, camera_id)] = transform

        return intrinsics, extrinsics

    @staticmethod
    def get_camera_parameters(frame, camera_id, intrinsics, extrinsics):
        """
        (frame, camera_id)에 해당하는 카메라 파라미터를 반환합니다.
        """
        intrinsic_params = intrinsics.get((frame, camera_id))
        extrinsic_matrix = extrinsics.get((frame, camera_id))
        return intrinsic_params, extrinsic_matrix

    @staticmethod
    def get_projection_matrix(frame, camera_id, intrinsics, extrinsics):
        """
        (frame, camera_id)에 해당하는 3x4 투영 행렬을 계산하여 반환합니다.
        """
        intrinsic_params, extrinsic_matrix = KITTIVideoDataset.get_camera_parameters(
            frame, camera_id, intrinsics, extrinsics
        )
        if intrinsic_params is None or extrinsic_matrix is None:
            return None

        fx, fy, cx, cy = intrinsic_params
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]])
        RT = extrinsic_matrix[:3, :]
        P = K @ RT  # 3x4 투영행렬
        return P

    def __getitem__(self, idx):
        """
        하나의 비디오 클립을 로드하여 (rgb_clip, depth_clip, masks) 또는
        검증 시에는 (rgb_clip, depth_clip, masks, extrinsics_list, intrinsics_list)를 반환합니다.
        """
        video_info = self.video_infos[idx]
        rgb_path = video_info['rgb_path']
        depth_path = video_info['depth_path']
        camera_idx = video_info['camera']
        intrinsic_file = video_info['intrinsic_file']
        extrinsic_file = video_info['extrinsic_file']

        rgb_files = sorted(os.listdir(rgb_path))
        depth_files = sorted(os.listdir(depth_path))

        if len(rgb_files) != len(depth_files):
            raise ValueError(f"RGB와 Depth의 개수가 일치하지 않습니다: {rgb_path}, {depth_path}")

        num_frames = len(rgb_files)
        if num_frames < self.clip_len:
            raise ValueError(f"비디오에 {self.clip_len}프레임이 존재하지 않습니다. 실제 프레임 수: {num_frames}")

        # 랜덤하게 clip_len 길이만큼 연속된 구간 선택
        start_idx = random.randint(0, num_frames - self.clip_len)

        # 첫 번째 RGB 프레임으로부터 crop 좌표 결정
        first_rgb_path = os.path.join(rgb_path, rgb_files[start_idx])
        first_rgb = Image.open(first_rgb_path).convert("RGB")
        resized_first = TF.resize(first_rgb, self.resize_size)
        crop_i, crop_j, crop_h, crop_w = get_random_crop_params(resized_first, self.resize_size)

        rgb_clip = []
        depth_clip = []
        masks = []
        frame_indices = []
        true_depth_clip = []

        # 카메라 파라미터 로드 (두 개의 딕셔너리 반환)
        intrinsics_dict, extrinsics_dict = KITTIVideoDataset.load_camera_params(
            intrinsic_file, extrinsic_file
        )

        # 카메라 파라미터를 미리 행렬 형태로 변환
        extrinsics_matrices = []  # [clip_len, 4, 4] 형태로 저장
        intrinsics_matrices = []  # [clip_len, 3, 3] 형태로 저장

        for i in range(self.clip_len):
            frame_idx = start_idx + i

            # 파일명에서 숫자 부분(“depth_000123.png” 또는 “rgb_000123.jpg”)만 추출
            depth_name = depth_files[frame_idx]
            # "depth_000123.png" → split('_')[-1] = "000123.png" → splitext → "000123"
            frame_num = int(os.path.splitext(depth_name.split('_')[-1])[0])
            frame_indices.append(frame_num)

            # RGB 처리
            rgb_name = rgb_files[frame_idx]
            rgb_img = Image.open(os.path.join(rgb_path, rgb_name)).convert("RGB")
            rgb_resized = TF.resize(rgb_img, self.resize_size)
            rgb_cropped = TF.crop(rgb_resized, crop_i, crop_j, crop_h, crop_w)
            rgb_tensor = TF.to_tensor(rgb_cropped)
            rgb_tensor = TF.normalize(rgb_tensor, mean=self.rgb_mean, std=self.rgb_std)
            rgb_clip.append(rgb_tensor)

            # Depth + Mask 처리
            disparity_img, mask_img, depth_m = self.load_depth_image_with_mask(
                os.path.join(depth_path, depth_name)
            )
            depth_resized = TF.resize(disparity_img, self.resize_size)
            depth_cropped = TF.crop(depth_resized, crop_i, crop_j, crop_h, crop_w)
            depth_tensor = TF.to_tensor(depth_cropped)

            mask_resized = TF.resize(mask_img, self.resize_size, interpolation=Image.NEAREST)
            mask_cropped = TF.crop(mask_resized, crop_i, crop_j, crop_h, crop_w)
            mask_tensor = torch.from_numpy(np.array(mask_cropped)).float().unsqueeze(0)
            
            depth_m_img = Image.fromarray(depth_m)           # float32→PIL
            depth_m_resized = TF.resize(depth_m_img, self.resize_size)
            depth_m_cropped = TF.crop(depth_m_resized, crop_i, crop_j, crop_h, crop_w)
            
            true_depth_tensor = torch.from_numpy(np.array(depth_m_cropped)).float().unsqueeze(0)  # [1, H, W]
            true_depth_clip.append(true_depth_tensor)  # 1채널 깊이(m)

            depth_clip.append(depth_tensor)
            masks.append(mask_tensor)

            # 카메라 파라미터 조회 및 행렬 변환
            intr_params, extr_matrix = KITTIVideoDataset.get_camera_parameters(
                frame_num, camera_idx, intrinsics_dict, extrinsics_dict
            )
            
            # Extrinsic 4x4 행렬 처리
            if extr_matrix is not None:
                extr_tensor = torch.tensor(extr_matrix, dtype=torch.float32)  # [4, 4]
            else:
                # 기본 단위 행렬
                extr_tensor = torch.eye(4, dtype=torch.float32)
                print(f"경고: 프레임 {frame_num}, 카메라 {camera_idx}에 대한 extrinsic 파라미터가 없습니다.")
            
            extrinsics_matrices.append(extr_tensor)
            
            # Intrinsic 3x3 행렬 처리
            if intr_params is not None:
                fx, fy, cx, cy = intr_params
                K = torch.tensor([
                    [fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0]
                ], dtype=torch.float32)  # [3, 3]
            else:
                # 기본 카메라 내부 파라미터
                K = torch.tensor([
                    [725.0087, 0.0, 620.5],
                    [0.0, 725.0087, 187.0],
                    [0.0, 0.0, 1.0]
                ], dtype=torch.float32)
                print(f"경고: 프레임 {frame_num}, 카메라 {camera_idx}에 대한 intrinsic 파라미터가 없습니다.")
            
            intrinsics_matrices.append(K)

        rgb_clip_tensor = torch.stack(rgb_clip)         # [clip_len, 3, H, W]
        depth_clip_tensor = torch.stack(depth_clip)     # [clip_len, 3, H, W]
        masks_tensor = torch.stack(masks)               # [clip_len, 1, H, W]
        true_depth_tensor = torch.stack(true_depth_clip)        

        if self.split == "train":
            return rgb_clip_tensor, depth_clip_tensor, masks_tensor
        else:
            # 카메라 파라미터를 텐서로 변환
            extrinsics_tensor = torch.stack(extrinsics_matrices)  # [clip_len, 4, 4]
            intrinsics_tensor = torch.stack(intrinsics_matrices)  # [clip_len, 3, 3]
            
            return rgb_clip_tensor, depth_clip_tensor, masks_tensor,  true_depth_tensor, extrinsics_tensor, intrinsics_tensor
