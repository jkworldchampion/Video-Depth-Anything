import os
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml
import wandb
import gc


from torch.utils.data import DataLoader 
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from utils.loss_MiDas import Loss_ssi, Loss_tgm
#from utils.loss_MiDas import MaskedScaleShiftInvariantLoss
from data.VKITTI import KITTIVideoDataset
from data.Google_Landmark import GoogleLandmarksDataset, CombinedDataset
from video_depth_anything.video_depth import VideoDepthAnything
from benchmark.eval.metric import *
from benchmark.eval.eval_tae import tae_torch
from PIL import Image

kitti_path = "/home/icons/workspace/SungChan/Video-Depth-Anything/datasets/KITTI"

# 2) 학습/검증 데이터셋 생성
kitti_train = KITTIVideoDataset(
    root_dir=kitti_path,
    clip_len=20,
    resize_size=518,
    split="train"
)
kitti_val = KITTIVideoDataset(
    root_dir=kitti_path,
    clip_len=20,
    resize_size=518,
    split="val"
)

print(len(kitti_train))