# Copyright (2025) Bytedance Ltd. and/or its affiliates 

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
import numpy as np

def compute_scale_and_shift(prediction, target, mask, scale_only=False):
    if scale_only:
        return compute_scale(prediction, target, mask), 0
    else:
        return compute_scale_and_shift_full(prediction, target, mask)


def compute_scale(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    prediction = prediction.astype(np.float32)
    target = target.astype(np.float32)
    mask = mask.astype(np.float32)

    a_00 = np.sum(mask * prediction * prediction)
    a_01 = np.sum(mask * prediction)
    a_11 = np.sum(mask)

    # right hand side: b = [b_0, b_1]
    b_0 = np.sum(mask * prediction * target)

    x_0 = b_0 / (a_00 + 1e-6)

    return x_0

def compute_scale_and_shift_full(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    prediction = prediction.astype(np.float32)
    target = target.astype(np.float32)
    mask = mask.astype(np.float32)

    a_00 = np.sum(mask * prediction * prediction)
    a_01 = np.sum(mask * prediction)
    a_11 = np.sum(mask)

    b_0 = np.sum(mask * prediction * target)
    b_1 = np.sum(mask * target)

    x_0 = 1
    x_1 = 0

    det = a_00 * a_11 - a_01 * a_01

    if det != 0:
        x_0 = (a_11 * b_0 - a_01 * b_1) / det
        x_1 = (-a_01 * b_0 + a_00 * b_1) / det

    return x_0, x_1


def get_interpolate_frames(frame_list_pre, frame_list_post):
    assert len(frame_list_pre) == len(frame_list_post)
    min_w = 0.0
    max_w = 1.0
    step = (max_w - min_w) / (len(frame_list_pre)-1)
    post_w_list = [min_w] + [i * step for i in range(1,len(frame_list_pre)-1)] + [max_w]
    interpolated_frames = []
    for i in range(len(frame_list_pre)):
        interpolated_frames.append(frame_list_pre[i] * (1-post_w_list[i]) + frame_list_post[i] * post_w_list[i])
    return interpolated_frames


def metric_val(infs, gts, poses, Ks):
    """
    least square 때문에, i & i+1 계산보다는 클립 하나를 통째로 계산하는 것이 올바름 
    infs,gts : [clip_len, ~] 꼴을 기대, inf는 

    지금 문제 : gt가 여기서는 진짜 gt를 기대하고있음 clipping 하기 전 ..
    
    LiheYoung씨가 rel-model인 경우에는 relative depth로 metric 해도 된다고 했음 
    -> 일단 gt말고 disparity랑 least square 때리기

    """
    ### 1. preprocessing
    valid_mask = (gts > 1e-3) & (gts < MAX_DEPTH)
    
    gt_disp_masked = 1. / (gts[valid_mask].reshape((-1,1)).double() + 1e-8)
    infs = infs.clamp(min=1e-3)
    pred_disp_masked = infs[valid_mask].reshape((-1,1)).double()
    
    _ones = torch.ones_like(pred_disp_masked)
    A = torch.cat([pred_disp_masked, _ones], dim=-1) 
    X = torch.linalg.lstsq(A, gt_disp_masked).solution  
    scale = X[0].item()
    shift = X[1].item()
    aligned_pred = scale * infs + shift
    aligned_pred = torch.clamp(aligned_pred, min=1e-3) 

    depth = torch.zeros_like(aligned_pred)
    depth = 1.0 / aligned_pred
    
    gt_depth = gts
    
    pred_depth = depth

    ### 4. validity
    n = valid_mask.sum((-1, -2))    ## 인풋이 3차원이었으므로 T차원임
    #print("n.shape: ", n.shape)
    valid_frame = (n > 0)   # 어 이거 torch.Size([32, 518, 518]) -> 32, ? 꼴인데 지금 ?
    pred_depth = pred_depth[valid_frame] # ok 가능함
    gt_depth = gt_depth[valid_frame]
    valid_mask = valid_mask[valid_frame]
    
    absrel = abs_relative_difference(pred_depth, gt_depth, valid_mask)
    delta1 = delta1_acc(pred_depth, gt_depth, valid_mask)
    tae = eval_tae(pred_depth, gt_depth, poses, Ks, valid_mask)

    
    return absrel,delta1,tae


def eval_tae(pred_depth, gt_depth, poses, Ks, masks):
    error_sum = 0.
    print("len_pred_depth : ",len(pred_depth))
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

# 극단값 마스킹
def get_mask(depth_m, min_depth, max_depth):
    valid_mask = (depth_m > min_depth) & (depth_m < max_depth)
    return valid_mask.bool()


def norm_ssi(depth, valid_mask):
    eps=1e-7
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
