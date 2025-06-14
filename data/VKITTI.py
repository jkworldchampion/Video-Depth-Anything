import os
import random
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

# val용 conter crop
def get_center_crop_params(img, output_size):
    w, h = img.size
    th, tw = output_size, output_size
    # resized한 img는 정사각형이므로 그냥 center_crop 좌표 = (0,0)
    # 하지만 안전하게 일반식 사용
    i = (h - th) // 2
    j = (w - tw) // 2
    return i, j, th, tw

def get_random_crop_params_with_rng(img, output_size, rng):
    w, h = img.size
    th, tw = output_size, output_size
    if w == tw and h == th:
        return 0, 0, th, tw
    i = rng.randint(0, h - th)
    j = rng.randint(0, w - tw)
    return i, j, th, tw


class KITTIVideoDataset(Dataset):
    def __init__(self,
                 root_dir,
                 clip_len=32,
                 resize_size=518,
                 split="train",
                 seed=0,
                 rgb_mean=(0.485, 0.456, 0.406),
                 rgb_std=(0.229, 0.224, 0.225)):
        super().__init__()
        assert split in ["train", "val"], "split은 'train' 또는 'val'이어야 합니다."

        self.clip_len = clip_len
        self.resize_size = resize_size
        self.split = split
        self.seed = seed
        self.epoch = 0
        self.root_dir = root_dir
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std
        self.IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

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
        
    def set_epoch(self, epoch: int):
        """
        매 epoch 시작 시 호출하세요.
        예) for epoch in range(num_epochs):
                 dataset.set_epoch(epoch)
                 for batch in loader: …
        """
        self.epoch = epoch

    def __len__(self):
        return len(self.video_infos)

    def load_depth(self, path):
        """
        16-bit PNG -> cm -> m 변환 후, PIL "F" 모드 이미지로 반환
        """
        depth_png = Image.open(path)
        depth_cm  = np.array(depth_png, dtype=np.uint16).astype(np.float32)
        depth_m   = depth_cm / 100.0
        return Image.fromarray(depth_m, mode="F")  # F mode가 뭘까


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
    def get_camera_parameters(frame, cam_id, intrinsics, extrinsics):
        return intrinsics.get((frame,cam_id)), extrinsics.get((frame,cam_id))

    
    @staticmethod
    def get_projection_matrix(frame, cam_id, intrinsics, extrinsics):
        intr, ext = KITTIVideoDataset.get_camera_parameters(frame, cam_id, intrinsics, extrinsics)
        if intr is None or ext is None:
            return None
        fx,fy,cx,cy = intr
        K  = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],dtype=np.float32)
        RT = ext[:3,:]
        return K @ RT
    
    def is_image_file(self, filename):
        return filename.lower().endswith(self.IMG_EXTENSIONS)
    
    def __getitem__(self, idx):
        info = self.video_infos[idx]
        
        # 폴더 명에서 .ipynb_checkpoint같은 거 없이 이미지만 잘
        rgb_files   = sorted([
            f for f in os.listdir(info['rgb_path'])
            if self.is_image_file(f) and os.path.isfile(os.path.join(info['rgb_path'], f))
        ])
        depth_files = sorted([
            f for f in os.listdir(info['depth_path'])
            if self.is_image_file(f) and os.path.isfile(os.path.join(info['depth_path'], f))
        ])
        # clip sampling
        N = len(rgb_files)
        
        # 1) RNG 초기화: 매 epoch마다, 매 idx마다 다른 시드
        if self.split == "train":
            rng = random.Random(self.seed + idx + self.epoch)
            start = rng.randint(0, N - self.clip_len)
        else:  # val: deterministic
            start = (N - self.clip_len) // 2
        
        # 2) crop 좌표 결정
        first_rgb = Image.open(os.path.join(info['rgb_path'], rgb_files[start])).convert("RGB")
        first_rgb = TF.resize(first_rgb, self.resize_size)
        if self.split == "train":
            ci, cj, ch, cw = get_random_crop_params_with_rng(first_rgb, self.resize_size, rng)
        else:
            ci, cj, ch, cw = get_center_crop_params(first_rgb, self.resize_size)

        rgb_seq, depth_seq = [], []
        # 카메라 파라미터 불러오기
        intrinsics_dict, extrinsics_dict = self.load_camera_params(info['intrinsic_file'], info['extrinsic_file'])
        extrinsics_list, intrinsics_list = [], []

        for i in range(self.clip_len):
            # ---- RGB 처리 ----
            img = Image.open(os.path.join(info['rgb_path'], rgb_files[start+i])).convert("RGB")
            img = TF.resize(img, self.resize_size)
            if self.split == "train":
                img = TF.crop(img, ci, cj, ch, cw)
            else:
                img = TF.center_crop(img, self.resize_size)
            t = TF.to_tensor(img)
            rgb_seq.append(TF.normalize(t, mean=self.rgb_mean, std=self.rgb_std))

            # ---- Depth 처리 ----
            d_img = self.load_depth(os.path.join(info['depth_path'], depth_files[start+i]))
            d_img = TF.resize(d_img, self.resize_size)
            if self.split == "train":
                d_img = TF.crop(d_img, ci, cj, ch, cw)
            else:
                d_img = TF.center_crop(d_img, self.resize_size)
            depth_seq.append(TF.to_tensor(d_img))

            # ---- Val일 때만 카메라 파라미터 쌓기 ----
            if self.split == "val":
                frame_num = int(depth_files[start+i].split('_')[-1].split('.')[0])
                intr_p, ext_m = self.get_camera_parameters(
                    frame_num, info['camera'], intrinsics_dict, extrinsics_dict
                )

                # extrinsic: numpy → Tensor
                if ext_m is None:
                    extrinsics_list.append(torch.eye(4, dtype=torch.float32))
                else:
                    extrinsics_list.append(torch.tensor(ext_m, dtype=torch.float32))

                # intrinsic: 파라미터 → 3×3 Tensor
                if intr_p is None:
                    fx, fy, cx, cy = 725.0087, 725.0087, 620.5, 187.0
                else:
                    fx, fy, cx, cy = intr_p
                K = torch.tensor([
                    [fx,   0.0, cx],
                    [0.0,  fy,  cy],
                    [0.0,  0.0, 1.0]
                ], dtype=torch.float32)
                intrinsics_list.append(K)

        rgb_tensor   = torch.stack(rgb_seq)    # [T,3,H,W]
        depth_tensor = torch.stack(depth_seq)  # [T,1,H,W]

        if self.split=="train":
            return rgb_tensor, depth_tensor
        else:
            extrinsics_tensor = torch.stack(extrinsics_list)
            intrinsics_tensor = torch.stack(intrinsics_list)
            return rgb_tensor, depth_tensor, extrinsics_tensor, intrinsics_tensor
