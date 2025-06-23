#! /usr/bin/python3
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
from utils.loss_MiDas import *
from data.Google_Landmark import GoogleLandmarksDataset, CombinedDataset
from video_depth_anything.video_depth import VideoDepthAnything
from benchmark.eval.metric import *
from benchmark.eval.eval_tae import tae_torch
from PIL import Image
from data.dataLoader import *

import logging

os.makedirs("logs", exist_ok=True)

# 2. configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[
        logging.StreamHandler(),                      # console
        logging.FileHandler("logs/train_log_task2.txt"),    # file
    ],
)

logger = logging.getLogger(__name__)

matplotlib.use('Agg')

MAX_DEPTH=80.0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MEAN = torch.tensor((0.485,0.456,0.406), device=DEVICE).view(3,1,1)
STD  = torch.tensor((0.229,0.224,0.225), device=DEVICE).view(3,1,1)

def least_sqaure_whole_clip(infs,gts):
    
    # 채널 없애고 [B,T,H,W] 로 만들기
    if infs.dim() == 5 and infs.shape[2] == 1:
        infs = infs.squeeze(2)
    if gts.dim() == 5 and gts.shape[2] == 1:
        gts  = gts .squeeze(2)

    ### 1. preprocessing
    valid_mask = (gts > 1e-3) & (gts < MAX_DEPTH)
    
    gt_disp_masked = 1. / (gts[valid_mask].reshape((-1,1)).double() + 1e-6)
    infs = infs.clamp(min=1e-3)
    pred_disp_masked = infs[valid_mask].reshape((-1,1)).double()
    
    ### 2. least square
    
    #print("valid_mask shape ( expecting 32 x H x W ) : ",valid_mask.shape)   
    #print("num valid pixels : ",valid_mask.sum())    ## 32개의 합이겠다. 
    
    #print("pred_disp_masked : ",pred_disp_masked)
    #print("pred_disp_masked.shape : ",pred_disp_masked.shape)
    #print("disparity_gts_disp_masked : ",disparity_gts_disp_masked)
    #print("disparity_gts_disp_masked.shape : ",disparity_gts_disp_masked.shape)
    
    _ones = torch.ones_like(pred_disp_masked)
    A = torch.cat([pred_disp_masked, _ones], dim=-1) 
    X = torch.linalg.lstsq(A, gt_disp_masked).solution  
    scale = X[0].item()
    shift = X[1].item()

    """

    x, y = pred_disp_masked, gt_disp_masked            # [N,1]
    mx, my = x.mean(), y.mean()
    cov  = ((x-mx)*(y-my)).sum()
    var  = ((x-mx)**2).sum()
    scale = (cov/var).item()
    shift = (my - scale*mx).item()
    """
    aligned_pred = scale * infs + shift
    aligned_pred = torch.clamp(aligned_pred, min=1e-3)  ## 근데 왜 1/80 아니고 .,,,???
    #aligned_pred = aligned_pred[valid_mask].view(-1, 1).double()  # [N, 1] 꼴로 맞추기
    #print("scale : ",scale)
    #print("shift : ",shift)
    #print("aligned_pred.shape : ",aligned_pred.shape)    ## 예상대로라면, [~,1] 느낌이어야함 아 이게 아니네

    depth = torch.zeros_like(aligned_pred)
    depth = 1.0 / aligned_pred
    
    pred_depth = depth
    
    return pred_depth, valid_mask



def metric_val(infs, gts, poses, Ks):
    
    gt_depth = gts
    pred_depth, valid_mask = least_sqaure_whole_clip(infs,gts)
    
    ### 4. validity
    n = valid_mask.sum((-1, -2))    ## 인풋이 3차원이었으므로 T차원임
    #print("n.shape: ", n.shape)
    valid_frame = (n > 0)   # 어 이거 torch.Size([32, 518, 518]) -> 32, ? 꼴인데 지금 ?
    #print("pred_depth.shape: ", pred_depth.shape)
    #print("valid_frame.shape : ",valid_frame.shape)
    pred_depth = pred_depth[valid_frame] # ok 가능함
    gt_depth = gt_depth[valid_frame]
    valid_mask = valid_mask[valid_frame]
    
    #print("valid_mask : ",valid_mask)
    #print("valid_frame", valid_frame)
    
    absrel = abs_relative_difference(pred_depth, gt_depth, valid_mask)
    delta1 = delta1_acc(pred_depth, gt_depth, valid_mask)
    tae = eval_tae(pred_depth, gt_depth, poses, Ks, valid_mask)

    
    return absrel,delta1,tae



def eval_tae(pred_depth, gt_depth, poses, Ks, masks):
    
    error_sum = 0.
    #print("len_pred_depth : ",len(pred_depth))
    for i in range(len(pred_depth) - 1):
        depth1 = pred_depth[i]
        depth2 = pred_depth[i+1]
        
        mask1 = masks[i]
        mask2 = masks[i+1]

        T_1 = poses[i]
        T_2 = poses[i+1]

        T_2_1 = torch.linalg.inv(T_2) @ T_1
   
        R_2_1 = T_2_1[:3,:3]
        t_2_1 = T_2_1[:3, 3]
        K = Ks[i]


        error1 = tae_torch(depth1, depth2, R_2_1, t_2_1, K, mask2)
        T_1_2 = torch.linalg.inv(T_2_1)
        R_1_2 = T_1_2[:3,:3]
        t_1_2 = T_1_2[:3, 3]


        error2 = tae_torch(depth2, depth1, R_1_2, t_1_2, K, mask1)
        
        error_sum += error1
        error_sum += error2
    
    result = error_sum / (2 * (len(pred_depth) -1))
    return result

"""

def eval_tae(pred_depth, gt_depth, poses, Ks, masks):
    error_sum = 0.
    B = len(pred_depth)
    for i in range(B - 1):
        depth1, depth2 = pred_depth[i], pred_depth[i+1]
        mask1, mask2 = masks[i],     masks[i+1]
        T1,   T2     = poses[i],     poses[i+1]
        K = Ks[i]

        R1, t1 = T1[:3,:3], T1[:3,3]    # [3,3], [3]
        R2, t2 = T2[:3,:3], T2[:3,3]
        R2_inv = R2.transpose(0,1)      # R2^T
        t2_inv = - R2_inv @ t2          # -R2^T t2

        R_2_1 = R2_inv @ R1             # [3,3]
        t_2_1 = R2_inv @ (t1 - t2)      # [3]

        error1 = tae_torch(depth1, depth2, R_2_1, t_2_1, K, mask2)

        R_1_2 = R_2_1.transpose(0,1)    # (R₂₁)^T
        t_1_2 = - R_1_2 @ t_2_1         # -R₂₁^T t₂₁
        error2 = tae_torch(depth2, depth1, R_1_2, t_1_2, K, mask1)

        error_sum += error1 + error2

    return error_sum / (2 * (B - 1))

"""


def get_mask(depth_m, min_depth, max_depth):
    valid_mask = (depth_m > min_depth) & (depth_m < max_depth)
    return valid_mask.bool()


def norm_ssi(depth, valid_mask):
    eps=1e-6
    disparity = torch.zeros_like(depth)
    disparity[valid_mask] = 1.0 / depth[valid_mask]

    # 이거 마스크 씌우면 자동으로 펼쳐지니까 일단 내가 shape 가져가기
    B, T, C, H, W = disparity.shape
    disp_flat = disparity.view(B, T, -1)         # [B, T, H*W]
    mask_flat = valid_mask.view(B, T, -1)       # [B, T, H*W]

    # 마스크 빼고 민맥스 값 찾기
    disp_min = disp_flat.masked_fill(~mask_flat, float('inf')).min(dim=-1)[0]
    disp_max = disp_flat.masked_fill(~mask_flat, float('-inf')).max(dim=-1)[0]

    disp_min = disp_min.view(B, T, 1, 1, 1)
    disp_max = disp_max.view(B, T, 1, 1, 1)

    denom = (disp_max - disp_min + eps)
    norm_disp = (disparity - disp_min) / denom

    # 걍 invalid는 0으로 만들기
    norm_disp = norm_disp.masked_fill(~valid_mask, 0.0)

    return norm_disp
    
# clip 단위의 scale-shift를 위해
def norm_ssi_clip(depth, valid_mask, eps=1e-6):
    """
    Clip-level min/max normalization of disparity for SSI loss.
    depth:   [B, T, 1, H, W]  (meters)
    valid_mask: same shape boolean
    returns norm_disp: [B, T, 1, H, W] in [0,1]
    """
    # 1) compute raw disparity
    disp = torch.zeros_like(depth)
    disp[valid_mask] = 1.0 / depth[valid_mask]

    B, T, C, H, W = disp.shape
    # 2) flatten per sample across all T×H×W
    disp_flat = disp.view(B, -1)              # [B, P]
    mask_flat = valid_mask.view(B, -1)        # [B, P]

    # 3) per-clip min and max (ignoring invalid pixels)
    disp_min = disp_flat.masked_fill(~mask_flat, float('inf')).min(dim=1)[0]  # [B]
    disp_max = disp_flat.masked_fill(~mask_flat, float('-inf')).max(dim=1)[0] # [B]

    # 4) reshape back to [B,1,1,1,1] so we can broadcast
    disp_min = disp_min.view(B, 1, 1, 1, 1)
    disp_max = disp_max.view(B, 1, 1, 1, 1)

    # 5) normalize entire clip with the same min/max
    norm = (disp - disp_min) / (disp_max - disp_min + eps)
    # 6) zero out invalid again
    norm = norm.masked_fill(~valid_mask, 0.0)
    return norm

    
def train(args):

    ### 0. prepare GPU, wandb_login
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=api_key)


    ### 1. Handling hyper_params with WAND :)
    config_path = "configs/config.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    hyper_params = config["hyper_parameter"]

    lr = hyper_params["learning_rate"]
    ratio_ssi = hyper_params["ratio_ssi"]
    ratio_tgm = hyper_params["ratio_tgm"]
    ratio_ssi_image = hyper_params["ratio_ssi_image"]
    num_epochs = hyper_params["epochs"]
    patient = hyper_params["patient"]
    batch_size = hyper_params["batch_size"]
    CLIP_LEN = hyper_params["clip_len"]
    #conv_out_channel = hyper_params["conv_out_channel"] 
    #conv = hyper_params["conv"]
    diff = args.diff
    hyper_params["diff"] = args.diff
    
    if args.diff :
        conv = args.conv
        hyper_params["conv"] = args.conv
        
        conv_out_channel = args.conv_out_channel
        hyper_params["conv_out_channel"] = args.conv_out_channel
        
        filename = f"diff_model_with_conv_{conv}_{conv_out_channel}.pth"
        
    else :
        conv_out_channel=0
        conv = False
        filename = "basic_model.pth"

    
    run = wandb.init(project="Temporal_Diff_Flow_experiment", entity="Depth-Finder", config=hyper_params)
    
    ### 2. Load data

    kitti_path = "/workspace/Video-Depth-Anything/datasets/KITTI"
    google_path="/workspace/Video-Depth-Anything/datasets/google_landmarks"

    rgb_clips, depth_clips = get_data_list(
        root_dir=kitti_path,
        data_name="kitti",
        split="train",
        clip_len=CLIP_LEN
    )

    kitti_train = KITTIVideoDataset(
        rgb_paths=rgb_clips,
        depth_paths=depth_clips,
        resize_size=518,
        split="train"
    )

    val_rgb_clips, val_depth_clips, val_cam_ids, val_intrin_clips, val_extrin_clips = get_data_list(
        root_dir=kitti_path,
        data_name="kitti",
        split="val",
        clip_len=CLIP_LEN
    )

    kitti_val = KITTIVideoDataset(
        rgb_paths=val_rgb_clips,
        depth_paths=val_depth_clips,
        cam_ids=val_cam_ids,
        intrin_clips=val_intrin_clips,
        extrin_clips=val_extrin_clips,
        resize_size=518,
        split="val"
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

    train_dataset = CombinedDataset(kitti_train,google_train,ratio=2)   
                    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=6)
    val_loader   = DataLoader(kitti_val,   batch_size=1, shuffle=False, num_workers=6)

    ### 3. Model and additional stuffs,...

    model = VideoDepthAnything(num_frames=CLIP_LEN,out_channels=[48, 96, 192, 384],out_channel=conv_out_channel,conv=conv,diff=diff).to(device)
    """
    out_channel : result of diff-conv
    conv : usage. False to use raw RGB diff
    """
    
    # 2) 체크포인트 파일 로드
    #model.load_state_dict(torch.load(f'video_depth_anything_vits.pth', map_location='cpu'), strict=False)
    #model = model.to(device).eval()
        
    # freeze -> pretrain은 DINO밖에 없어서 이렇게 가능 
    for param in model.pretrained.parameters():
        param.requires_grad = False

    # multi GPUs setting
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params}")

    loss_tgm = LossTGMVector(static_th=0.05)
    loss_ssi = Loss_ssi()

    wandb.watch(model, log="all")

    best_val_loss = float('inf')
    best_epoch = 0
    trial = 0

    scaler = GradScaler()

    torch.backends.cuda.preferred_linalg_library('cusolver')

    ### 4. train
    
    logger.info(f"train_loader_len: {len(train_loader)}")

    start_epoch = 0
    for epoch in tqdm(range(start_epoch, num_epochs), desc="Epoch", leave=False):
        print()
        epoch_loss = 0.0
        
        model.train()

        for batch_idx, ((x, y),(x_image, y_image)) in tqdm(enumerate(train_loader)):
            
            ## 마스크 생성
            video_masks = get_mask(y,min_depth=0.001,max_depth=80.0)
            x, y = x.to(device), y.to(device)
            video_masks = video_masks.to(device)
            
            # point. image는 B랑 T랑 바꿔줘야 계산할때 연속하지 않다고 판단할 수 있을 듯
            x_image = x_image.permute(1,0,2,3,4)
            y_image = y_image.permute(1,0,2,3,4)
            
            img_masks = get_mask(y_image,min_depth=1/80.0, max_depth= 1000.0)
            x_image, y_image = x_image.to(device), y_image.to(device)
            img_masks = img_masks.to(device)
            
            #print("x.shape",x.shape) # [16, 1, 3, 518, 518]
            #print("y.shape",y.shape) # [16, 1, 1, 518, 518]
            #print("video_masks.shape",video_masks.shape) # [16, 1, 1, 518, 518]
            
            optimizer.zero_grad()
            with autocast():
                pred = model(x)  
                #masks_squeezed = video_masks.squeeze(2)  # [B, T, H, W]
                #print("pred: ", pred)
                #print("gt :", y)
                print("")
                logger.info(f"video : pred.mean(): {pred.mean().item():.6f}")
                #print("image : pred.mean():", pred_image.mean().item())
                #print("valid_mask.sum():", masks_squeezed.sum().item())
                #print("y.sum():", y.sum().item())
                #if pred.sum()== 0:
                #print("pred_sum = 0, see GT : ",y[0][0])
                
                pred_image = model(x_image)
                logger.info(f"image : pred.mean(): {pred_image.mean().item():.6f}")
            
                
                #print("pred.isnan().any():", torch.isnan(pred).any().item())
                #print("pred.isinf().any():", torch.isinf(pred).any().item())
                #if pred.sum()==0:
                #print("pred_sum = 0, see GT : ",y[0][0])

                # 유효 픽셀에만 곱하기
                #pred_image_masked = pred_image * mask_image
                #y_image_masked    = y_image    * mask_image

                #loss_tgm_val = loss_tgm(pred_masked, y_masked, masks_squeezed)
                
                # 이거 norm_ssi 갔다오면 disp로 줌 
                # disp_normed = norm_ssi(y,video_masks)

                # clip-level SSI normalization
                disp_normed = norm_ssi_clip(y, video_masks)
                video_masks_squeezed = video_masks.squeeze(2)
                loss_ssi_value = loss_ssi(pred, disp_normed, video_masks_squeezed)   ## 어차피 5->4 는 loss에서 해줌 
                
                
                #disp_normed = norm_ssi(y,video_masks)
                #infs_fit,_ = least_sqaure_whole_clip(infs=pred,gts=y)    # 이렇게 나온 결과는 depth임 
                
                #print("pred: ", pred[0][0])
                #print("infs_fit: ", infs_fit[0][0])
                #print("y: ", y[0][0])
                loss_tgm_value = loss_tgm(pred, y, video_masks_squeezed)
                
                # =============== single img =================== # 
                img_disp_normed = norm_ssi(y_image,img_masks)
                img_masks_squeezed = img_masks.squeeze(2)
                loss_ssi_image_value = loss_ssi(pred_image, img_disp_normed, img_masks_squeezed)


                # =============== pred.mean loss 계산 =============== #
                gt_disp = (1.0 / y.clamp(min=1e-6)).squeeze(2)    # [B, T, H, W]
                mean_pred = pred.mean()
                mean_gt   = gt_disp.mean()
                loss_mean = torch.abs(mean_pred - mean_gt)
                
                #if epoch < 5 : 
                #    loss = ratio_ssi * loss_ssi_value + ratio_ssi_image * loss_ssi_image_value
                
                #else :
                loss = ratio_tgm * loss_tgm_value + ratio_ssi * loss_ssi_value + ratio_ssi_image * loss_ssi_image_value
                # loss += 0.01 * loss_mean  # mean loss 추가
                
                #print(">> loss.shape:", loss.shape)
                #print("check ",loss_tgm_value,loss_ssi_value,loss_ssi_image_value)
                #loss = loss_tgm_val * 0.1 + loss_ssi_val
                #loss = loss_ssi_val 
            
            # 또는 스케일링 방식 사용: 유효한 픽셀 수로 정규화
            # valid_pixel_ratio = masks.sum() / (masks.shape[0] * masks.shape[1] * masks.shape[2] * masks.shape[3] * masks.shape[4])
            # loss = (ratio_tgm * loss_tgm_val + ratio_ssi * loss_ssi_val) / (valid_pixel_ratio + 1e-8)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            #loss.backward()
            #optimizer.step()
            
            epoch_loss += loss.item()

            if batch_idx % 5 == 0:
                logger.info(f"Epoch [{epoch}], Batch [{batch_idx}], Loss: {loss.item():.4f}")
            
        avg_train_loss = epoch_loss / len(train_loader)
        
        scheduler.step()

        logger.info(f"Epoch [{epoch}/{num_epochs}] Train Loss: {avg_train_loss:.4f}")    
        
        # === validation loop ===
        model.eval()
        val_loss = 0.0
        total_absrel = 0.0
        total_delta1 = 0.0
        total_tae = 0.0
        cnt_clip = 0
        
        with torch.no_grad():
            for batch_idx, (x, y, extrinsics, intrinsics) in tqdm(enumerate(val_loader)):
                # 1) move to device
                x, y = x.to(device), y.to(device)
                extrinsics, intrinsics = extrinsics.to(device), intrinsics.to(device)

                # 2) model inference + basic losses
                pred = model(x)                                        # [B, T, H, W]
                masks = get_mask(y, min_depth=0.001, max_depth=80.0)   # [B, T, 1, H, W]
                masks = masks.to(device).bool()
                disp_normed   = norm_ssi(y, masks)
                ssi_loss_val  = loss_ssi(pred, disp_normed, masks.squeeze(2))
                tgm_loss_val  = loss_tgm(pred, y, masks)
                val_loss     += ratio_ssi * ssi_loss_val + ratio_tgm * tgm_loss_val

                logger.info(f"pred.mean(): {pred.mean().item():.6f}")

                # 3) prepare for scale & shift
                B, T, H, W = pred.shape

                MIN_DISP = 1.0 / 80.0      # depth=80m → disp=0.0125
                MAX_DISP = 1.0 / 0.001     # depth=0.001m → disp=1000.0

                raw_disp = pred.clamp(min=1e-6)                # [B, T, H, W]
                gt_disp  = (1.0 / y.clamp(min=1e-6)).squeeze(2) # [B, T, H, W]
                m_flat   = masks.squeeze(2).view(B, -1).float()# [B, P]
                p_flat   = raw_disp.view(B, -1)               # [B, P]
                g_flat   = gt_disp .view(B, -1)               # [B, P]

            
                # 4) build A, b for least-squares: A @ [a; b] ≈ b_vec 
                A = torch.stack([p_flat, torch.ones_like(p_flat, device=device)], dim=-1)  # [B,P,2]
                A = A * m_flat.unsqueeze(-1)    
                b_vec = g_flat.unsqueeze(-1) * m_flat.unsqueeze(-1)  # mask out invalid
                
                 # 5) batched least-squares
                X = torch.linalg.lstsq(A, b_vec).solution  # [B,2,1]
                a = X[:,0,0].view(B,1,1,1)                 # [B,1,1,1]
                b = X[:,1,0].view(B,1,1,1)                 # [B,1,1,1]

                aligned_disp = (raw_disp * a + b).clamp(min=MIN_DISP, max=MAX_DISP)  # [B,T,H,W]

                """
                AtA = torch.matmul(A.transpose(-2, -1), A)
                Atb = torch.matmul(A.transpose(-2, -1), b_vec)# [B,P,1]
                X = torch.linalg.solve(AtA, Atb)
                
                a = X[:, 0, 0].view(B, 1, 1, 1)  # [B,1,1,1]
                b = X[:, 1, 0].view(B, 1, 1, 1)  # [B,1,1,1]
                
                aligned_disp = (raw_disp * a + b).clamp(min=MIN_DISP, max=MAX_DISP)
                

                # 2) 유효 픽셀 개수
                count = m_flat .sum(dim=1)                   # [B]
                
                # 3) 평균 계산
                sum_x = (p_flat  * m_flat).sum(dim=1)             # [B]
                sum_y = (g_flat  * m_flat).sum(dim=1)             # [B]
                mx    = sum_x / count                  # [B]
                my    = sum_y / count                  # [B]
                
                # 4) 공분산·분산 계산 (unnormalized)
                cov = ((p_flat - mx.unsqueeze(1)) * (g_flat - my.unsqueeze(1)) * m_flat ).sum(dim=1)   # [B]
                var = ((p_flat - mx.unsqueeze(1))**2 * m_flat ).sum(dim=1)                       # [B]
                
                # 5) scale & shift
                scale = cov / (var + 1e-6)             # [B]  (eps로 나눗셈 안정화)
                shift = my - scale * mx                # [B]
                
                # 6) reshape back to broadcastable
                a = scale.view(B, 1, 1, 1)             # [B,1,1,1]
                b = shift.view(B, 1, 1, 1)             # [B,1,1,1]
                
                # 7) 최종 정렬된 disparity
                aligned_disp = (raw_disp * a + b).clamp(min=MIN_DISP, max=MAX_DISP)  # [B, T, H, W]

                """
                
                # 4) 첫 배치에만 프레임 저장
                if batch_idx == 0:
                    save_dir = f"outputs/task2/diff_{diff}_conv_{conv}_ch_{conv_out_channel}/epoch_{epoch}_batch_{batch_idx}"
                    os.makedirs(save_dir, exist_ok=True)
                    wb_images = []  # W&B 에 보낼 이미지 리스트
                    for t in range(T):

                        # a) RGB
                        rgb_norm = x[0, t]  # [3,H,W]
                        rgb_unc  = (rgb_norm * STD + MEAN).clamp(0,1)
                        rgb_np   = (rgb_unc.cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
                        Image.fromarray(rgb_np).save(os.path.join(save_dir, f"rgb_{t:02d}.png"))

                        # b) GT Disparity 저장 (Min–Max 정규화)
                        depth_frame = y[0, t].squeeze(0).clamp(min=1e-6)       # [H,W]
                        disp_frame  = 1.0 / depth_frame                       # [H,W]
                        valid       = masks[0, t].squeeze(0)                  # [H,W] bool

                        # 유효 픽셀만 뽑아 min/max
                        d_vals = disp_frame[valid]
                        d_min, d_max = d_vals.min(), d_vals.max()

                        norm_gt = (disp_frame - d_min) / (d_max - d_min + 1e-6)
                        norm_gt = norm_gt.clamp(0,1)

                        gt_uint8 = (norm_gt.cpu().numpy() * 255).astype(np.uint8)
                        gt_rgb   = np.stack([gt_uint8]*3, axis=-1)
                        Image.fromarray(gt_rgb).save(os.path.join(save_dir, f"gt_{t:02d}.png"))

                        # c) Mask 저장
                        mask_frame = masks[0, t].squeeze(0).cpu().numpy().astype(np.uint8) * 255
                        Image.fromarray(mask_frame).save(os.path.join(save_dir, f"mask_{t:02d}.png"))
                        
                        # d) Predicted Disparity 저장 (같은 Min–Max 사용)
                        pred_frame = aligned_disp[0, t]  # [H,W]
                        norm_pd = (pred_frame - d_min) / (d_max - d_min + 1e-6)
                        norm_pd = norm_pd.clamp(0,1)

                        pd_uint8 = (norm_pd.cpu().numpy() * 255).astype(np.uint8)
                        pd_rgb   = np.stack([pd_uint8]*3, axis=-1)
                        Image.fromarray(pd_rgb).save(os.path.join(save_dir, f"pred_{t:02d}.png"))
                        
                        # e) pred-disparity wandb에 저장
                        wb_images.append(wandb.Image(os.path.join(save_dir, f"pred_{t:02d}.png"), caption=f"pred_epoch{epoch}_frame{t:02d}"))

                    logger.info(f"→ saved validation frames to '{save_dir}'")

                # 5) metric 평가 (모든 배치에 대해)
                for b in range(B):
                    inf_clip  = pred[b]              # [T,H,W]
                    gt_clip   = y[b].squeeze(1)      # [T,H,W]
                    mask_clip = masks[b].squeeze(1)  # [T,H,W]
                    pose      = extrinsics[b]
                    Kmat      = intrinsics[b]
                    absr, d1, tae = metric_val(inf_clip, gt_clip, pose, Kmat)
                    total_absrel  += absr
                    total_delta1  += d1
                    total_tae     += tae
                    cnt_clip     += 1

            # 최종 통계
            avg_val_loss = val_loss / len(val_loader)
            avg_absrel   = total_absrel / cnt_clip
            avg_delta1   = total_delta1 / cnt_clip
            avg_tae      = total_tae / cnt_clip

        logger.info(f"Epoch [{epoch}/{num_epochs}] Validation Loss: {avg_val_loss:.4f}")
        logger.info(f"AbsRel  : {avg_absrel:.4f}")
        logger.info(f"Delta1  : {avg_delta1:.4f}")
        logger.info(f"TAE    : {avg_tae:.4f}")

        wandb.log({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "absrel": avg_absrel,
            "delta1": avg_delta1,
            "tae": avg_tae,
            "epoch": epoch,
            "pred_disparity": wb_images,
        })
        
        ### best 체크포인트 저장
        is_best = False
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch+1
            is_best = True
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),  # mixed training
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'trial': trial,
            }, filename)
            logger.info(f"Best checkpoint saved at epoch {epoch+1} with validation loss {avg_val_loss:.4f}")
            trial = 0
        else:
            trial += 1

        """
        if trial >= patient:
            print("Early stopping triggered.")
            break
            
        """
    # 최종 모델 저장

    logger.info(f"Training finished. Best checkpoint was from epoch {best_epoch} with validation loss {best_val_loss:.4f}.")
    run.finish()

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--diff",action="store_true")
    parser.add_argument("--conv",action="store_true")
    parser.add_argument("--conv_out_channel", type=int, default=0)
    
    args = parser.parse_args()
    train(args)
