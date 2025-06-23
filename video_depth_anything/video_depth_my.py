import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose
import cv2
from tqdm import tqdm
import numpy as np
import gc

from .dinov2 import DINOv2
from .dpt_temporal import DPTHeadTemporal
from .util.transform import Resize, NormalizeImage, PrepareForNet

from utils.util import compute_scale_and_shift, get_interpolate_frames
from timm.models.layers import DropPath
from torchvision.ops import DeformConv2d



# Placeholder for a Deformable Multi-Head Attention
class DeformableCrossFrameAttention(nn.Module):
    """
    Deformable cross-frame attention with dynamic sampling points.
    """
    def __init__(self, dim, num_heads=8, max_points=8):
        super().__init__()
        self.num_heads = num_heads
        self.max_points = max_points
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        # predict offsets for max_points
        self.offset_pred = nn.Linear(dim, num_heads * max_points * 2)
        self.attn_out = nn.Linear(dim, dim)

    def forward(self, query, reference, spatial_size, n_points=None):
        # query, reference: [B, N, C], N = H*W
        if n_points is None:
            n_points = self.max_points
        else:
            n_points = min(n_points, self.max_points)

        B, N, C = query.shape
        H, W = spatial_size

        # 1) q, k, v 투영
        q = self.q_proj(query).view(B, N, self.num_heads, C // self.num_heads)
        kv = self.kv_proj(reference).view(B, N, 2, self.num_heads, C // self.num_heads)
        k, v = kv[:, :, 0], kv[:, :, 1]  # 각각 [B, N, heads, head_dim]

        # 2) 오프셋 예측 후 slice
        offsets = self.offset_pred(query).view(B, N, self.num_heads, self.max_points, 2)
        offsets = offsets[:, :, :, :n_points]  # [B, N, heads, n_points, 2]

        # 3) base grid 생성 (indexing='ij' 권장)
        ys = torch.arange(H, device=query.device)
        xs = torch.arange(W, device=query.device)
        coords = torch.stack(torch.meshgrid(ys, xs, indexing='ij'), -1)  # [H, W, 2]
        coords = coords.view(1, N, 1, 1, 2).float()  # [1, N, 1, 1, 2]

        # 4) sampling locations 계산 및 정규화
        sample_locs = coords + offsets  # [B, N, heads, n_points, 2]
        sample_locs[..., 0] = sample_locs[..., 0] / (H - 1) * 2 - 1
        sample_locs[..., 1] = sample_locs[..., 1] / (W - 1) * 2 - 1

        # 5) batch 차원으로 확장하여 grid 준비
        #    -> grid: [B*heads*n_points, H, W, 2]
        grid = sample_locs.view(B, self.num_heads * n_points, H, W, 2)
        grid = grid.reshape(B * self.num_heads * n_points, H, W, 2)

        # 6) reference k 맵을 batch 차원으로 확장
        #    -> ref_k_exp: [B*heads*n_points, C, H, W]
        ref_k = reference.view(B, C, H, W)
        ref_k_exp = ref_k.unsqueeze(1).expand(-1, self.num_heads * n_points, -1, -1, -1)
        ref_k_exp = ref_k_exp.reshape(B * self.num_heads * n_points, C, H, W)

        # 7) grid_sample로 키/값 샘플링
        sampled = F.grid_sample(ref_k_exp, grid, align_corners=True)  # [B*Hn, C, H, W]

        # 8) 다시 [B, heads, n_points, head_dim, N] 형태로 재구성
        sampled = sampled.view(
            B, self.num_heads, n_points, C // self.num_heads, H * W
        )  # [B, heads, n_points, head_dim, N]

        # 9) 어텐션 계산을 위해 [B, N, heads, n_points, head_dim]으로 변환
        sampled = sampled.permute(0, 4, 1, 2, 3)  # [B, N, heads, n_points, head_dim]
        sampled_k = sampled
        sampled_v = sampled

        # 10) q 차원 맞추기
        q = q.unsqueeze(3)  # [B, N, heads, 1, head_dim]

        # 11) 어텐션 스코어 및 가중합
        attn = (q * sampled_k).sum(-1) * self.scale  # [B, N, heads, n_points]
        attn = attn.softmax(-1)
        out = (attn.unsqueeze(-1) * sampled_v).sum(-2)  # [B, N, heads, head_dim]
        out = out.flatten(2)  # [B, N, C]

        return self.attn_out(out)  # [B, N, C]

class VideoDepthAnything(nn.Module):
    def __init__(
        self,
        encoder='vits',
        features=64,
        out_channels=[48, 96, 192, 384],
        use_bn=False,
        use_clstoken=False,
        num_frames=32,
        pe='ape',
        num_block=3,
        out_channel=64,
        conv=True,
        diff=True
    ):
        super().__init__()

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23]
        }
        self.encoder = encoder
        # encoder & head
        self.pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.head = DPTHeadTemporal(
            self.pretrained.embed_dim, features, use_bn,
            out_channels=out_channels, use_clstoken=use_clstoken,
            num_frames=num_frames, pe=pe
        )
        # diff + conv
        self.diff, self.conv = diff, conv
        self.out_channel = out_channel
        if diff:
            if conv:
                self.diff_layers = nn.Sequential(
                    nn.Conv2d(3, out_channel//2, 3, padding=1), nn.BatchNorm2d(out_channel//2), nn.ReLU(True),
                    nn.Conv2d(out_channel//2, out_channel, 3, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(True)
                )
            else:
                self.out_channel = 3
            self.mlp = nn.Conv2d(self.out_channel, self.pretrained.embed_dim, 1)
        # temporal modules
        self.cross_attn = DeformableCrossFrameAttention(
            dim=self.pretrained.embed_dim, num_heads=8, max_points=8
        )
        self.skip_conv = nn.Conv2d(1, self.pretrained.embed_dim, 3, padding=1)
        # store previous depth
        self.prev_depth = None

    def forward(self, x):
        B, T, C, H, W = x.shape
        ph, pw = H//14, W//14
        # encoder feats
        feats = self.pretrained.get_intermediate_layers(
            x.flatten(0,1),
            self.intermediate_layer_idx[self.encoder],
            return_class_token=True
        )
        # diff pipeline
        if self.diff:
            dt = torch.stack([x[:,1:] - x[:,:-1]],0).squeeze(0)
            dt_flat = dt.view(-1,C,H,W)
            if self.conv:
                dt_flat = self.diff_layers(dt_flat)
            pooled = F.adaptive_avg_pool2d(dt_flat,(ph,pw)).view(B*(T-1), self.out_channel, ph, pw)
            df = self.mlp(pooled).view(B, T-1, -1, ph, pw)
            df_list = []
            for i in range(T):
                if i==0: df_list.append(df[:,0])
                elif i==T-1: df_list.append(df[:,-1])
                else: df_list.append(0.5*(df[:,i-1]+df[:,i]))
            df_tokens = torch.stack(df_list,1).permute(0,1,3,4,2).reshape(B*T, ph*pw, -1)
            updated = [(f+df_tokens, cls) for (f,cls) in feats]
        else:
            updated = feats
        # temporal deformable attention linear decrease of n_points
        updated_feats = []
        for (f_tok, cls) in updated:
            Bn, N, Cdim = f_tok.shape
            f_bt = f_tok.view(B, T, N, Cdim)
            fused = []
            for t in range(T):
                q = f_bt[:,t].reshape(-1,N,Cdim)
                attn_sum = 0
                count = 0
                # over all frames
                for r in range(T):
                    d = abs(t-r)
                    # linear n_points: max at d=0, min 1 at d=T-1
                    n_p = max(1, int(self.cross_attn.max_points * (1 - d/(T-1))))
                    ref = f_bt[:,r].reshape(-1,N,Cdim)
                    attn_sum += self.cross_attn(q, ref, (ph,pw), n_points=n_p)
                    count += 1
                fused.append(attn_sum / count)
            ft = torch.stack(fused,1)
            # skip connection
            if self.prev_depth is not None:
                pd = F.interpolate(self.prev_depth.view(B*T,1,H,W),(ph,pw),align_corners=True,mode='bilinear')
                pd_feat = self.skip_conv(pd).view(B, T, ph*pw, Cdim)
                ft = ft + pd_feat
            updated_feats.append((ft.reshape(B*T,N,Cdim), cls))
        # decode & update prev
        # depth = self.head(updated_feats, ph, pw, T)
        # self.prev_depth = depth.detach()
        # out = F.interpolate(depth.view(B*T,1,ph,pw),(H,W),mode='bilinear',align_corners=True)
        # return F.relu(out).view(B,T,H,W)
        depth = self.head(updated_feats, ph, pw, T)                        # [B*T, ph, pw]
        depth = depth.unsqueeze(1)
        depth = F.interpolate(depth, (H, W), mode='bilinear',align_corners=True)         # [B*T,1,H,W]
        return depth.view(B,T,H,W)

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
        
