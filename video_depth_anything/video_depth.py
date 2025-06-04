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

EMB_DIM = 384

class VideoDepthAnything(nn.Module):
    def __init__(
        self,
        encoder='vitl',
        features=64, 
        out_channels=[256, 512, 1024, 1024], 
        use_bn=False, 
        use_clstoken=False,
        num_frames=32,
        pe='ape',
        num_block=3,
        out_channel=64,
        conv=True
    ):
        super(VideoDepthAnything, self).__init__()

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23]
        }
        
        self.encoder = encoder
        #self.pretrained = DINOv2(model_name=encoder)
        self.pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')


        self.head = DPTHeadTemporal(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken, num_frames=num_frames, pe=pe)
        ###
        
        """
        
        self.conv = conv
        self.out_channel = out_channel
        
        if self.conv :
            self.diff_layers = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=out_channel//2, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channel//2),
                nn.ReLU(True),
    
                nn.Conv2d(in_channels=out_channel//2, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True)
            )
        else :
            self.out_channel = 3 # rgb
    
            
        self.mlp = nn.Conv2d(self.out_channel, EMB_DIM, kernel_size=1)
        
        """

    def forward(self, x):
        #print("x.shape :",x.shape)
        B, T, C, H, W = x.shape
        patch_h, patch_w = H // 14, W // 14
        features = self.pretrained.get_intermediate_layers(x.flatten(0,1), self.intermediate_layer_idx[self.encoder], return_class_token=True)

        #print("f1",features[0][0].shape)
        """
        diff = []

        for idx in range(B):
            temp = []
            for k,k_next in zip(x[idx][:-1], x[idx][1:]):
                temp.append(k_next-k)
            diff.append(temp)

        diff_tensor = torch.stack([torch.stack(frames, dim=0) for frames in diff],dim=0)    # 결과는 B, T-1, C , H , W
        diff_flat = diff_tensor.view(-1, C, H, W)

        #print("diff_flat",diff_flat[0])
        #print("diff_flat_shape",diff_flat.shape)

        if self.conv:
            for layer in self.diff_layers:
                diff_flat = layer(diff_flat)

        #print("diff_flat after conv",diff_flat.shape)
        
        pooled = F.adaptive_avg_pool2d(diff_flat,(patch_h, patch_w)) ## cnn으로도 가능 .. 
        pooled = pooled.view(B*(T-1), self.out_channel, patch_h, patch_w)
        #pooled = pooled.permute(0, 1, 3, 4, 2)  ## B T-1  H W C

        #print("pooled",pooled.shape)

        pooled= self.mlp(pooled) #  B T-1 384 H W 

        #print("pooled",pooled.shape)
        
        pooled = pooled.view(B, (T-1), EMB_DIM, patch_h, patch_w)
        
        diff_per_frame = []
        for i in range(T):
            if i == 0:
                # 첫번째 프레임
                diff_per_frame.append(pooled[:, 0, :, :, :])  
            elif i == T - 1:
                # 마지막 프레임
                diff_per_frame.append(pooled[:, T - 2, :, :, :])
            else:
                # 중간 프레임
                avg_ij = 0.5 * (pooled[:, i - 1, :, :, :] + pooled[:, i, :, :, :])
                diff_per_frame.append(avg_ij)
            
        # B T 384 H W 
 
        diff_all = torch.stack(diff_per_frame, dim=1)
        
        diff_result = diff_all.permute(0, 1, 3, 4, 2)  ## B T H W C
        diff_result = diff_result.view(B*T,patch_h*patch_w,EMB_DIM)

        #print("diff_result", diff_result.shape)

        new_feat = []
        updated_features = []
        for feat,cls in features:
            new_feat.append(feat + diff_result)

        for feat, (_,cls) in zip(new_feat, features):
            updated_features.append((feat, cls))


        """
        depth = self.head(features, patch_h, patch_w, T)
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=True)
        depth = F.relu(depth)
        return depth.squeeze(1).unflatten(0, (B, T)) # return shape [B, T, H, W]
    
    def infer_video_depth(self, frames, target_fps, input_size=518, device='cuda', fp32=False):
        frame_height, frame_width = frames[0].shape[:2]
        ratio = max(frame_height, frame_width) / min(frame_height, frame_width)
        if ratio > 1.78:  # we recommend to process video with ratio smaller than 16:9 due to memory limitation
            input_size = int(input_size * 1.777 / ratio)
            input_size = round(input_size / 14) * 14

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

        frame_list = [frames[i] for i in range(frames.shape[0])]
        frame_step = INFER_LEN - OVERLAP
        org_video_len = len(frame_list)
        append_frame_len = (frame_step - (org_video_len % frame_step)) % frame_step + (INFER_LEN - frame_step)
        frame_list = frame_list + [frame_list[-1].copy()] * append_frame_len
        
        depth_list = []
        pre_input = None
        for frame_id in tqdm(range(0, org_video_len, frame_step)):
            cur_list = []
            for i in range(INFER_LEN):
                cur_list.append(torch.from_numpy(transform({'image': frame_list[frame_id+i].astype(np.float32) / 255.0})['image']).unsqueeze(0).unsqueeze(0))
            cur_input = torch.cat(cur_list, dim=1).to(device)
            if pre_input is not None:
                cur_input[:, :OVERLAP, ...] = pre_input[:, KEYFRAMES, ...]

            with torch.no_grad():
                with torch.autocast(device_type=device, enabled=(not fp32)):
                    depth = self.forward(cur_input) # depth shape: [1, T, H, W]

            depth = depth.to(cur_input.dtype)
            depth = F.interpolate(depth.flatten(0,1).unsqueeze(1), size=(frame_height, frame_width), mode='bilinear', align_corners=True)
            depth_list += [depth[i][0].cpu().numpy() for i in range(depth.shape[0])]

            pre_input = cur_input

        del frame_list
        gc.collect()

        depth_list_aligned = []
        ref_align = []
        align_len = OVERLAP - INTERP_LEN
        kf_align_list = KEYFRAMES[:align_len]

        for frame_id in range(0, len(depth_list), INFER_LEN):
            if len(depth_list_aligned) == 0:
                depth_list_aligned += depth_list[:INFER_LEN]
                for kf_id in kf_align_list:
                    ref_align.append(depth_list[frame_id+kf_id])
            else:
                curr_align = []
                for i in range(len(kf_align_list)):
                    curr_align.append(depth_list[frame_id+i])
                scale, shift = compute_scale_and_shift(np.concatenate(curr_align),
                                                       np.concatenate(ref_align),
                                                       np.concatenate(np.ones_like(ref_align)==1))

                pre_depth_list = depth_list_aligned[-INTERP_LEN:]
                post_depth_list = depth_list[frame_id+align_len:frame_id+OVERLAP]
                for i in range(len(post_depth_list)):
                    post_depth_list[i] = post_depth_list[i] * scale + shift
                    post_depth_list[i][post_depth_list[i]<0] = 0
                depth_list_aligned[-INTERP_LEN:] = get_interpolate_frames(pre_depth_list, post_depth_list)

                for i in range(OVERLAP, INFER_LEN):
                    new_depth = depth_list[frame_id+i] * scale + shift
                    new_depth[new_depth<0] = 0
                    depth_list_aligned.append(new_depth)

                ref_align = ref_align[:1]
                for kf_id in kf_align_list[1:]:
                    new_depth = depth_list[frame_id+kf_id] * scale + shift
                    new_depth[new_depth<0] = 0
                    ref_align.append(new_depth)
            
        depth_list = depth_list_aligned
            
        return np.stack(depth_list[:org_video_len], axis=0), target_fps
        
