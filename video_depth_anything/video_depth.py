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
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import Compose
import cv2
from tqdm import tqdm
import numpy as np
import gc

from .dinov2 import DINOv2
from .dpt_temporal import DPTHeadTemporal
from .util.transform import Resize, NormalizeImage, PrepareForNet

from utils.util import compute_scale_and_shift, get_interpolate_frames

# infer settings, do not change
INFER_LEN = 32
OVERLAP = 10
KEYFRAMES = [0,12,24,25,26,27,28,29,30,31]
INTERP_LEN = 8

class VideoDepthAnything(nn.Module):
    def __init__(
        self,
        encoder='vitl',
        features=256, 
        out_channels=[256, 512, 1024, 1024], 
        use_bn=False, 
        use_clstoken=False,
        num_frames=32,
        pe='ape'
    ):
        super(VideoDepthAnything, self).__init__()

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23]
        }
        
        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)

        self.head = DPTHeadTemporal(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken, num_frames=num_frames, pe=pe)

    def forward(self, x):
        B, T, C, H, W = x.shape
        patch_h, patch_w = H // 14, W // 14
        features = self.pretrained.get_intermediate_layers(x.flatten(0,1), self.intermediate_layer_idx[self.encoder], return_class_token=True)
        depth = self.head(features, patch_h, patch_w, T)
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=True)
        depth = F.relu(depth)
        return depth.squeeze(1).unflatten(0, (B, T)) # return shape [B, T, H, W]
    
    def infer_video_depth(self, frames, target_fps, input_size=518, device='cuda', fp32=False):
        # frame 크기 조절
        frame_height, frame_width = frames[0].shape[:2]
        ratio = max(frame_height, frame_width) / min(frame_height, frame_width)
        if ratio > 1.78:  # we recommend to process video with ratio smaller than 16:9 due to memory limitation
            input_size = int(input_size * 1.777 / ratio)
            input_size = round(input_size / 14) * 14
        # preprocess settings
        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        
        # 원본 frame을 list로 변환
        frame_list = [frames[i] for i in range(frames.shape[0])]
        frame_step = INFER_LEN - OVERLAP  # step size for inference
        org_video_len = len(frame_list)   # padding을 추가하기 위해 원본 비디오 길이 저장
        append_frame_len = (frame_step - (org_video_len % frame_step)) % frame_step + (INFER_LEN - frame_step)
        frame_list = frame_list + [frame_list[-1].copy()] * append_frame_len  # 마지막 프레임을 추가하여 길이를 맞춤
        
        # sliding window inference
        depth_list = []
        pre_input = None
        for frame_id in tqdm(range(0, org_video_len, frame_step)):
            # 한 window (len=32)만큼 처리
            cur_list = []
            for i in range(INFER_LEN):
                cur_list.append(torch.from_numpy(transform({'image': frame_list[frame_id+i].astype(np.float32) / 255.0})['image']).unsqueeze(0).unsqueeze(0))
            cur_input = torch.cat(cur_list, dim=1).to(device)
            # 현재의 frame에서 이전 frame의 KEYFRAMES를 overlap으로 추가. 
            if pre_input is not None:
                cur_input[:, :OVERLAP, ...] = pre_input[:, KEYFRAMES, ...]
            # FP32 or FP16 중에 선택~
            with torch.no_grad():
                with torch.autocast(device_type=device, enabled=(not fp32)):
                    depth = self.forward(cur_input) # depth shape: [1, T, H, W]

            depth = depth.to(cur_input.dtype) # Support autocast inference, grayscale, NPZ and EXR output
            depth = F.interpolate(depth.flatten(0,1).unsqueeze(1), size=(frame_height, frame_width), mode='bilinear', align_corners=True)
            depth_list += [depth[i][0].cpu().numpy() for i in range(depth.shape[0])]
            
            # 현재의 frame을 다음 frame의 pre_input으로 저장
            pre_input = cur_input

        # 중간 데이터 정리
        del frame_list
        gc.collect()

        # 윈도우 간 scale and shift 정합&보간
        depth_list_aligned = []
        ref_align = []
        align_len = OVERLAP - INTERP_LEN  # 10 - 8 = 2
        kf_align_list = KEYFRAMES[:align_len]  # [0, 12]를 사용하여 align

        for frame_id in range(0, len(depth_list), INFER_LEN):  # 0, 32, 64, ...
            # 첫 번째 윈도우인지 확인
            if len(depth_list_aligned) == 0:
                # 첫 번째 윈도우는 그냥 추가
                depth_list_aligned += depth_list[:INFER_LEN]
                # 첫 번째 윈도우의 align을 위한 reference depth 설정, 즉, [0, 12]의 depth를 넣은 것
                for kf_id in kf_align_list:
                    ref_align.append(depth_list[frame_id+kf_id])
            else:
                # 두 번째 윈도우부터는 앞에서 가져온 [0, 12]를 활용해 scale and shift를 적용
                curr_align = []
                for i in range(len(kf_align_list)): # align_len = 2 이므로 0, 1
                    curr_align.append(depth_list[frame_id+i])  # 즉, 현재의 0, 1을 가져옴
                    # 왜 여기서 0, 12가 아닌 0, 1을 가져오는지?
                # scale and shift 계산
                scale, shift = compute_scale_and_shift(np.concatenate(curr_align),  # 이전의 0, 12와 
                                                       np.concatenate(np.ones_like(ref_align)==1))  # 현재의 0, 1과 ss 계산
                                                       np.concatenate(ref_align),
                # overlap된 10개 중 뒷부분 8개는 보간을 통해 채움
                pre_depth_list = depth_list_aligned[-INTERP_LEN:]
                post_depth_list = depth_list[frame_id+align_len:frame_id+OVERLAP]
                for i in range(len(post_depth_list)):
                    post_depth_list[i] = post_depth_list[i] * scale + shift
                    post_depth_list[i][post_depth_list[i]<0] = 0
                depth_list_aligned[-INTERP_LEN:] = get_interpolate_frames(pre_depth_list, post_depth_list)

                # 나머지 새 프레임(OVERLAP 이후 ~ 32) 추가, 즉 22개 frames
                for i in range(OVERLAP, INFER_LEN):
                    new_depth = depth_list[frame_id+i] * scale + shift  # 위에서 계산한 scale과 shift 적용
                    new_depth[new_depth<0] = 0  # masking
                    depth_list_aligned.append(new_depth)

                # ref_aling 갱신, 첫 번째 요소는 유지, 나머지 1개는 최근의 frame의 세번째 요소로 교체
                ref_align = ref_align[:1]  # 0번째 frame은 전역정보를 위해 유지
                for kf_id in kf_align_list[1:]: 
                    new_depth = depth_list[frame_id+kf_id] * scale + shift
                    new_depth[new_depth<0] = 0
                    ref_align.append(new_depth)
            
        depth_list = depth_list_aligned
            
        return np.stack(depth_list[:org_video_len], axis=0), target_fps
        