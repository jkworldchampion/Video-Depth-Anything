import os
import torch
import yaml
import numpy as np
from tqdm import tqdm
from PIL import Image

from torch.utils.data import DataLoader
from video_depth_anything.video_depth import VideoDepthAnything
from data.VKITTI import KITTIVideoDataset
from data.Google_Landmark import CombinedDataset
from utils.loss_MiDas import Loss_ssi, LossTGMVector
from benchmark.eval.metric import *
from utils.util import *

# 초기 설정
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MEAN = torch.tensor((0.485, 0.456, 0.406), device=DEVICE).view(3, 1, 1)
STD  = torch.tensor((0.229, 0.224, 0.225), device=DEVICE).view(3, 1, 1)
MIN_DISP = 1.0 / 80.0
MAX_DISP = 1.0 / 0.001

def validate_only():
    # 1. Config 불러오기
    with open("configs/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    hp = config["hyper_parameter"]

    # 2. Dataset 준비
    kitti_path = "/home/icons/workspace/SungChan/Video-Depth-Anything/datasets/KITTI"
    google_path = "/home/icons/workspace/SungChan/Video-Depth-Anything/datasets/google_landmarks"

    val_dataset = CombinedDataset(
        KITTIVideoDataset(kitti_path, hp["clip_len"], resize_size=518, seed=hp["seed"], split="val"),
        google_image_root=os.path.join(google_path, "images"),
        google_depth_root=os.path.join(google_path, "depth"),
        output_size=518,
        seed=hp["seed"]
    )
    val_loader = DataLoader(val_dataset, batch_size=hp["batch_size"], shuffle=False, num_workers=2, pin_memory=False)

    # 3. 모델 로딩 (vits or vitl)
    encoder = hp.get("encoder", "vits")  # encoder 항목이 없을 경우 기본값 'vits'
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    model = VideoDepthAnything(**model_configs[encoder]).to(DEVICE)
    ckpt_path = f"./video_depth_anything_{encoder}.pth"
    print(f"[INFO] Loading pretrained weights from: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE), strict=True)
    model.eval()

    # 4. Loss 및 Metric
    loss_ssi = Loss_ssi()
    loss_tgm = LossTGMVector(static_th=hp["threshold"])
    ratio_ssi = hp["ratio_ssi"]
    ratio_tgm = hp["ratio_tgm"]

    val_loss, total_absrel, total_delta1, total_tae, cnt_clip = 0, 0, 0, 0, 0

    with torch.no_grad():
        for batch_idx, (x, y, extrinsics, intrinsics) in tqdm(enumerate(val_loader)):
            x, y = x.to(DEVICE), y.to(DEVICE)
            extrinsics, intrinsics = extrinsics.to(DEVICE), intrinsics.to(DEVICE)

            pred = model(x)  # [B, T, H, W]
            masks = get_mask(y, min_depth=0.001, max_depth=80.0).to(DEVICE).bool()

            disp_normed = norm_ssi(y, masks)
            ssi_val = loss_ssi(pred, disp_normed, masks.squeeze(2))
            tgm_val = loss_tgm(pred, y, masks)
            val_loss += ratio_ssi * ssi_val + ratio_tgm * tgm_val

            # Disparity 정렬
            B, T, H, W = pred.shape
            raw_disp = pred.clamp(min=1e-6)
            gt_disp = (1.0 / y.clamp(min=1e-6)).squeeze(2)
            m_flat = masks.squeeze(2).view(B, -1).float()
            p_flat = raw_disp.view(B, -1)
            g_flat = gt_disp.view(B, -1)

            A = torch.stack([p_flat, torch.ones_like(p_flat)], dim=-1) * m_flat.unsqueeze(-1)
            b_vec = g_flat.unsqueeze(-1) * m_flat.unsqueeze(-1)
            X = torch.linalg.lstsq(A, b_vec).solution
            a = X[:, 0, 0].view(B, 1, 1, 1)
            b = X[:, 1, 0].view(B, 1, 1, 1)
            aligned_disp = (raw_disp * a + b).clamp(min=MIN_DISP, max=MAX_DISP)

            # 저장
            if batch_idx == 0:
                os.makedirs("outputs/frames_pretrained", exist_ok=True)
                for t in range(T):
                    rgb = (x[0, t] * STD + MEAN).clamp(0, 1)
                    rgb_np = (rgb.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    Image.fromarray(rgb_np).save(f"outputs/frames_pretrained/rgb_{t:02d}.png")

                    depth_frame = y[0, t].squeeze(0).clamp(min=1e-6)
                    disp_frame = 1.0 / depth_frame
                    valid = masks[0, t].squeeze(0)
                    d_vals = disp_frame[valid]
                    d_min, d_max = d_vals.min(), d_vals.max()

                    norm_gt = (disp_frame - d_min) / (d_max - d_min + 1e-6)
                    gt_uint8 = (norm_gt.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                    Image.fromarray(gt_uint8).save(f"outputs/frames_pretrained/gt_{t:02d}.png")

                    pred_frame = aligned_disp[0, t]
                    norm_pred = (pred_frame - d_min) / (d_max - d_min + 1e-6)
                    pd_uint8 = (norm_pred.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                    Image.fromarray(pd_uint8).save(f"outputs/frames_pretrained/pred_{t:02d}.png")

            # Metric 평가
            for b in range(B):
                absrel, delta1, tae = metric_val(pred[b], y[b].squeeze(1), extrinsics[b], intrinsics[b])
                total_absrel += absrel
                total_delta1 += delta1
                total_tae += tae
                cnt_clip += 1

    print("\n✅ Validation Summary")
    print(f"Val Loss : {val_loss / len(val_loader):.4f}")
    print(f"AbsRel   : {total_absrel / cnt_clip:.4f}")
    print(f"Delta1   : {total_delta1 / cnt_clip:.4f}")
    print(f"TAE      : {total_tae / cnt_clip:.4f}")


if __name__ == "__main__":
    validate_only()
