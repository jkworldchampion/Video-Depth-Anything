import os
import random
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from glob import glob

def get_random_crop_params_with_rng(img, output_size, rng):
    w, h = img.size
    th, tw = output_size, output_size
    if w == tw and h == th:
        return 0, 0, th, tw
    i = rng.randint(0, h - th)
    j = rng.randint(0, w - tw)
    return i, j, th, tw

    
def get_data_list(root_dir, data_name, split, clip_len=16, condition_num=10):
    """
    data path랑 clip_len, 원하는 데이터 개수 or idx 개수만큼 리스트로 쌓아주는 함수
    """
    if data_name == "kitti":
        if split == "train":
            video_info_train = get_kitti_video_path(root_dir, condition_num=condition_num, split="train", binocular=False)
            x_path, y_path = get_kitti_individuals(video_info_train, clip_len, split)
        
        else :
            video_info_val = get_kitti_video_path(root_dir, condition_num=condition_num, split="val", binocular=False)
            x_path, y_path,cam_ids, intrin_clips, extrin_clips = get_kitti_individuals(video_info_val, clip_len, split)
            
    elif data_name == "google":
        if split == "train":
            x_path, y_path = get_google_paths(root_dir)
        else:
            x_path, y_path = get_google_paths(root_dir)
        
        
    if split=="val":
        return x_path, y_path, cam_ids, intrin_clips, extrin_clips 
    else:
        return x_path, y_path

def get_google_paths(root_dir):

    exts = ['.png', '.jpg', '.npy']
    x_paths = []
    y_paths = []
    
    img_root = os.path.join(root_dir, "images")
    dep_root = os.path.join(root_dir, "depth")

    for folder in sorted(os.listdir(img_root)):
        img_dir = os.path.join(img_root, folder)
        dep_dir = os.path.join(dep_root, folder)
        if not os.path.isdir(img_dir):
            print(f"경고: 이미지 폴더 없음: {img_dir}")
            continue
        if not os.path.isdir(dep_dir):
            print(f"경고: depth 폴더 없음: {dep_dir}")
            continue

        for fname in sorted(os.listdir(img_dir)):
            img_path = os.path.join(img_dir, fname)
            base, _ = os.path.splitext(fname)
            for ext in exts:
                dep_path = os.path.join(dep_dir, base + ext)
                if os.path.isfile(dep_path):
                    x_paths.append(img_path)
                    y_paths.append(dep_path)
                    break
            else:
                print(f"경고: 대응하는 depth 파일 없음: {img_path}")

    return x_paths, y_paths



def get_kitti_paths(video_info, clip_len, split):
    x_clips = []
    y_clips = []
    intrin_clips = []
    extrin_clips = []
    cam_ids = []

    for info in video_info:
        rgb_dir        = info['rgb_path']
        depth_dir      = info['depth_path']
        intrinsic_file = info['intrinsic_file']
        extrinsic_file = info['extrinsic_file']
        camera_id      = info['camera']

        rgb_files   = sorted(os.listdir(rgb_dir))
        depth_files = sorted(os.listdir(depth_dir))
        if len(rgb_files) != len(depth_files):
            continue

        n = len(rgb_files)
        num_clips = n // clip_len

        for i in range(num_clips):
            start = i * clip_len
            end   = start + clip_len
            if end > n:
                end_idx.append()
                break

            x_clips.append([os.path.join(rgb_dir,   f) for f in rgb_files[start:end]])
            y_clips.append([os.path.join(depth_dir, f) for f in depth_files[start:end]])
            intrin_clips.append(intrinsic_file)
            extrin_clips.append(extrinsic_file)
            cam_ids.append(camera_id)

    #print("x clip count:", len(x_clips))
    #print("y clip count:", len(y_clips))
    
    if split == "train":
        return x_clips, y_clips 
    else:
        return x_clips, y_clips , cam_ids, intrin_clips, extrin_clips 
    


def get_kitti_individuals(video_info, clip_len, split):
    """
    기존  get_kitti_paths 의 문제점 : 클립단위로 넘겨주기 때문에 이걸 몇번째에서 자를 수 없음. 
    그대신 end index를 저장해두고, clip 단위가 아닌 전체 영상을 넘겨주기

    returns : 전체 영상 데이터 + scene의 끝점
    """
    x_clips = []
    y_clips = []
    intrin_clips = []
    extrin_clips = []
    cam_ids = []
    
    for info in video_info:
        rgb_dir        = info['rgb_path']
        depth_dir      = info['depth_path']
        intrinsic_file = info['intrinsic_file']
        extrinsic_file = info['extrinsic_file']
        camera_id      = info['camera']

        rgb_files   = sorted(os.listdir(rgb_dir))
        depth_files = sorted(os.listdir(depth_dir))
        if len(rgb_files) != len(depth_files):
            continue

        n = len(rgb_files) // clip_len # 즉 이제 n은 몫이에요

        x_clips.append([os.path.join(rgb_dir,f) for f in rgb_files[:n*clip_len]])
        y_clips.append([os.path.join(depth_dir,f) for f in depth_files[:n*clip_len]])
        intrin_clips.append(intrinsic_file)
        extrin_clips.append(extrinsic_file)
        cam_ids.append(camera_id)

    

    if split == "train":
        return x_clips, y_clips
    else:
        return x_clips, y_clips , cam_ids, intrin_clips, extrin_clips 
    
    
    
def get_kitti_video_path(root_dir, condition_num, split, binocular):
    """
    condition_num: 각 scene에서 몇 개의 condition을 가져올지
    """
    
    # 데이터 개수 ( 단안기준 )
    # scene 1 : 446 
    # scene 2 : 232
    # scene 3 : 269
    # scene 4 : 338
    # scene 5 : 836
    # => 만약 16씩 돌리면 80번 iter 돌아가면 끝남

    rgb_root = os.path.join(root_dir, "vkitti_2.0.3_rgb")
    depth_root = os.path.join(root_dir, "vkitti_2.0.3_depth")
    textgt_root = os.path.join(root_dir, "vkitti_2.0.3_textgt")
    
    video_infos = []

    for scene in sorted(os.listdir(rgb_root)):
        scene_rgb_path = os.path.join(rgb_root, scene)
        scene_depth_path = os.path.join(depth_root, scene)
        scene_textgt_path = os.path.join(textgt_root, scene)

        if not os.path.isdir(scene_rgb_path) or \
            not os.path.isdir(scene_depth_path) or \
            not os.path.isdir(scene_textgt_path):
            continue

        if (split == "train" and "Scene06" in scene) or \
            (split == "val" and "Scene06" not in scene):
            continue

        for idx, condition in enumerate(sorted(os.listdir(scene_rgb_path))):
            print(f"Processing scene: {scene}, condition: {condition}")
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
            
            if binocular:
                cam_paths = ["Camera_0", "Camera_1"] 
            else:
                cam_paths = ["Camera_0"]
                
            for cam in cam_paths:
                cam_idx = int(cam[-1])  # "Camera_0" → 0, "Camera_1" → 1
                rgb_path = os.path.join(cond_rgb_path, "frames", "rgb", cam)
                depth_path = os.path.join(cond_depth_path, "frames", "depth", cam)

                if os.path.isdir(rgb_path) and os.path.isdir(depth_path):
                    video_infos.append({
                        'rgb_path': rgb_path,
                        'depth_path': depth_path,
                        'intrinsic_file': intrinsic_file,
                        'extrinsic_file': extrinsic_file,
                        'scene': scene,
                        'condition': condition,
                        'camera': cam_idx
                    })
                    
            if idx == condition_num-1:
                break

    # 이제 video_infos에는 scene,condition,camera 따라서 경로가 설정됨
    
    return video_infos

class KITTIVideoDataset(Dataset):
    def __init__(
        self,
        rgb_paths,
        depth_paths,
        cam_ids=None,
        intrin_clips=None,
        extrin_clips=None,
        seed = 42,
        rgb_mean=(0.485, 0.456, 0.406),
        rgb_std=(0.229, 0.224, 0.225),
        resize_size=518,
        split="train",
        clip_len=16,
    ):
        super().__init__()
        assert split in ["train", "val"]
        assert len(rgb_paths) == len(depth_paths)
        self.rgb_paths = rgb_paths
        self.depth_paths = depth_paths
        self.intrin_clips  = intrin_clips
        self.extrin_clips  = extrin_clips
        self.cam_ids = cam_ids
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std
        self.resize_size = resize_size
        self.split = split
        self.seed = seed
        self.epoch = 0
        self.clip_len = clip_len

        # scene별로  effective clip 계산
        scene_clip_counts = [
            len(scene_rgb)//clip_len - 1  # 마지막 클립은 버리기 -> 오버플로 방지
            for scene_rgb in self.rgb_paths
        ]

        # 총 클립 개수
        self.total_clips = sum(scene_clip_counts)

        # flat idx -> scene_idx, chunk_idx
        self.flat2scene = [0] * self.total_clips
        self.flat2chunk = [0] * self.total_clips

        ptr = 0
        for scene_idx, n_eff in enumerate(scene_clip_counts):
            for chunk_idx in range(n_eff):
                self.flat2scene[ptr] = scene_idx
                self.flat2chunk[ptr] = chunk_idx
                ptr += 1
        

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.total_clips

    def load_depth(self, path):
        depth_png = Image.open(path)
        depth_cm = np.array(depth_png, dtype=np.uint16).astype(np.float32)
        depth_m = depth_cm / 100.0
        depth_img = Image.fromarray ((depth_m), mode="F") 

        return depth_img

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
        지금 여기에 들어온 rgb_paths들은 그냥 전체 데이터를 가지고 있음  -> 이중리스트로 가지고있음 scene별로
        end_idx로 이걸 적절히 핸들링 해야함
        현재 받아오는 idx는 전체 영상길이 / clip_len으로 할거임 -> 아 이러면 안되는게, scene별로 나눠지는 몫 기준으로 해야함 ㅇㅇ 그치
        무튼 일단 idx는 각 clip의 개수라고 생각해보자.

        정리해보자면, idx는 clip idx를 넣어야함. 즉 
        """

        ## algorithm : 결국 start index만 잘 뽑으면 해결되는 문제임
        ## 주의해야할 점은, 위에서 넘겨줄때 16으로 나눠 떨어지는거만 준게 아님.-> 이러면 문제가, 나머지가 각각 다르니까 그냥 clip len만큼으로 자르자. 수정했음
        ## shift는 0부터 15까지 가능.
        ## 64개 였다면, -> 이걸 나누기 16 하면 4개가 나오는데
        """
        remaining = idx ## 예를 들어 idx가 5이다. 즉 5번째 클립을 받는 타이밍이라고 해보자
        for scene_idx, scene_rgb in enumerate(self.rgb_paths):
            n_clips = len(scene_rgb)// self.clip_len    # 만약 1번째 씬이 64개라고 해보면, nclip = 4
            effective = n_clips - 1 # 마지막꺼 버림 이슈 for overflow 방지
            if remaining < effective:
                break
            remaining -= effective    # remaining = 2
        chunk_idx = remaining   # for문을 나오고 나면, chunk idx는 scene_idx에 해당하는 씬의 몇번째 클립인지가 됨

        -> 이거 오버헤드 너무큼 생각해보면. 위로 올리기
        """

                # 2) train split이면 여기서 바로 반환
        if self.split == "train":

            scene_idx = self.flat2scene[idx]
            chunk_idx = self.flat2chunk[idx]

            scene_rgb_paths   = self.rgb_paths[scene_idx]
            scene_depth_paths = self.depth_paths[scene_idx]

            rng = random.Random(self.seed + self.epoch)
            shift = rng.randint(0, self.clip_len-1)

            base = shift + chunk_idx * self.clip_len
            rgb_paths  = scene_rgb_paths[base:base+self.clip_len]
            depth_paths= scene_depth_paths[base:base+self.clip_len]

            #rgb_paths = self.rgb_clips[idx]
            #depth_paths = self.depth_clips[idx]

            first = Image.open(rgb_paths[0]).convert("RGB")
            first = TF.resize(first, self.resize_size)
            i, j, th, tw = get_random_crop_params_with_rng(first, self.resize_size, rng)
            
            rgb_seq, depth_seq = [], []
            for rp, dp in zip(rgb_paths, depth_paths):
                img = Image.open(rp).convert("RGB")
                img = TF.resize(img, self.resize_size)
                img = TF.crop(img, i, j, th, tw)
                img = TF.normalize(TF.to_tensor(img), mean=self.rgb_mean, std=self.rgb_std)
                rgb_seq.append(img)

                depth = self.load_depth(dp)
                depth = TF.resize(depth, self.resize_size)
                depth = TF.crop(depth, i, j, th, tw)
                depth_seq.append(TF.to_tensor(depth))

            rgb_tensor = torch.stack(rgb_seq)     # [clip_len, 3, H, W]
            depth_tensor = torch.stack(depth_seq) # [clip_len, 1, H, W]

            return rgb_tensor, depth_tensor


        else :

            scene_idx = self.flat2scene[idx]
            chunk_idx = self.flat2chunk[idx]
            
            scene_rgb_paths   = self.rgb_paths[scene_idx]
            scene_depth_paths = self.depth_paths[scene_idx]

            base = chunk_idx * self.clip_len
            rgb_paths  = scene_rgb_paths  [base:base+self.clip_len]
            depth_paths= scene_depth_paths[base:base+self.clip_len]

            rgb_seq, depth_seq = [], []
            for rp, dp in zip(rgb_paths, depth_paths):
                img = Image.open(rp).convert("RGB")
                img = TF.resize(img, self.resize_size)
                img = TF.center_crop(img, self.resize_size)
                img = TF.normalize(TF.to_tensor(img), mean=self.rgb_mean, std=self.rgb_std)
                rgb_seq.append(img)

                depth = self.load_depth(dp)
                depth = TF.resize(depth, self.resize_size)
                depth = TF.center_crop(depth, self.resize_size)
                depth_seq.append(TF.to_tensor(depth))

            rgb_tensor = torch.stack(rgb_seq)     # [clip_len, 3, H, W]
            depth_tensor = torch.stack(depth_seq) # [clip_len, 1, H, W]

            
            # 3) val split일 때만 카메라 파라미터 로딩
            camera_id = self.cam_ids[scene_idx]
            intrinsic_file = self.intrin_clips[scene_idx]
            extrinsic_file = self.extrin_clips[scene_idx]
            intrinsics_dict, extrinsics_dict = self.load_camera_params(intrinsic_file, extrinsic_file)

            extrinsics_list, intrinsics_list = [], []
            for dp in depth_paths:
                frame_num = int(os.path.splitext(os.path.basename(dp))[0].split('_')[-1])
                intr_p, extr_m = self.get_camera_parameters(frame_num, camera_id, intrinsics_dict, extrinsics_dict)

                if extr_m is None:
                    extr_m = np.eye(4, dtype=np.float32)
                extrinsics_list.append(torch.tensor(extr_m, dtype=torch.float32))

                if intr_p is None:
                    fx, fy, cx, cy = 725.0087, 725.0087, 620.5, 187.0
                else:
                    fx, fy, cx, cy = intr_p
                K = torch.tensor([[fx, 0.0, cx],
                                [0.0, fy, cy],
                                [0.0, 0.0, 1.0]], dtype=torch.float32)
                intrinsics_list.append(K)

            extrinsics_tensor = torch.stack(extrinsics_list)   # [clip_len, 4, 4]
            intrinsics_tensor = torch.stack(intrinsics_list)   # [clip_len, 3, 3]
            return rgb_tensor, depth_tensor, extrinsics_tensor, intrinsics_tensor

class GoogleDepthDataset(Dataset):
    def __init__(
        self,
        img_paths,
        depth_paths,
        rgb_mean=(0.485, 0.456, 0.406),
        rgb_std=(0.229, 0.224, 0.225),
        resize_size=518,
        seed=42
    ):
        super().__init__()
        assert len(img_paths) == len(depth_paths), "이미지/뎁스 개수 불일치"
        self.img_paths   = img_paths
        self.depth_paths = depth_paths
        self.resize_size = resize_size
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):

        rng = random.Random(self.seed + self.epoch)
        # RGB 이미지 

        first = Image.open(self.img_paths[0]).convert("RGB")
        first = TF.resize(first, self.resize_size)
        i, j, th, tw = get_random_crop_params_with_rng(first, self.resize_size, rng)

        img = Image.open(self.img_paths[idx]).convert("RGB")
        img = TF.resize(img, self.resize_size)
        img = TF.crop(img, i, j, th, tw)
        img = TF.normalize(TF.to_tensor(img),mean=self.rgb_mean,std=self.rgb_std)

        # Disparity/Depth
        dp = self.depth_paths[idx]
        if dp.endswith('.npy'):
            disp = np.load(dp).astype(np.float32)
            disp_img = Image.fromarray(disp)
        else:
            disp_img = Image.open(dp).convert("F")
        disp_img = TF.resize(disp_img, self.resize_size)
        disp_img = TF.crop(disp_img, i, j, th, tw)
        disp = torch.from_numpy(np.array(disp_img, np.float32)).unsqueeze(0)

        return img,disp

class CombinedDataset(Dataset):
    def __init__(self, kitti_dataset, google_dataset,ratio=4):
        super().__init__()
        self.ratio = ratio
        self.kitti_dataset = kitti_dataset
        self.google_dataset = google_dataset
        
        #print("kitti len ",len(self.kitti_dataset))
        #print("google len ",len(self.google_dataset) )

    def set_epoch(self, epoch):
        self.kitti_dataset.set_epoch(epoch)
        
    def __len__(self):
        ## 이거 비율 1:4로 주고싶어서 
        return min(len(self.kitti_dataset), len(self.google_dataset)// self.ratio)
    
    def __getitem__(self, idx):
        kitti_item = self.kitti_dataset[idx]
        start = idx * self.ratio
        google_items = [self.google_dataset[start + i] for i in range(self.ratio)]
        
        google_imgs  = torch.stack([item[0] for item in google_items], dim=0)  # [ratio, 3, H, W]
        google_depths = torch.stack([item[1] for item in google_items], dim=0)  # [ratio, 1, H, W]

        return kitti_item, (google_imgs,google_depths)

"""

from torch.utils.data import Dataset, DataLoader


kitti_path = "/workspace/Video-Depth-Anything/datasets/KITTI"
google_path="/workspace/Video-Depth-Anything/datasets/google_landmarks"

rgb_clips, depth_clips = get_data_list(
    root_dir=kitti_path,
    data_name="kitti",
    split="train",
    clip_len=16
)

kitti_train = KITTIVideoDataset(
    rgb_paths=rgb_clips,
    depth_paths=depth_clips,
    resize_size=518,
    split="train"
)

google_img_paths, google_depth_paths = get_data_list(
    root_dir=google_path,
    data_name="google",
    split="train"
)

google_train = GoogleDepthDataset(
    img_paths=google_img_paths,
    depth_paths=google_depth_paths,
    resize_size=518
)

train_dataset = CombinedDataset(kitti_train,google_train,ratio=4)   
                
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

import os
from PIL import Image
import torch

def dump_dataloader(loader, num_epochs=2):

    idx = 0
    clip_idx = 0
    for epoch in range(num_epochs):
        for batch_idx, ((rgb,depth),(_,_)) in enumerate(loader):
            # train이면 2-tuple, val이면 4-tuple
            # 배치 차원 제거 (여기선 batch_size=1 가정)
            # rgb: [B, T, 3, H, W], depth: [B, T, 1, H, W]
            B, T, C, H, W = rgb.shape
            rgb = rgb[0]    # [T,3,H,W]
            depth = depth[0]# [T,1,H,W]

            for t in range(T):
                idx+=1
                print("total frame : ",idx)

            clip_idx+=1
            print("total clip : ", clip_idx)



dump_dataloader(train_loader, num_epochs=2)

"""
