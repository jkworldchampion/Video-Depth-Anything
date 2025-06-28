import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose
from torch.utils.checkpoint import checkpoint
import cv2
from tqdm import tqdm
import numpy as np
import gc

from .dinov2 import DINOv2
from .dpt_temporal import DPTHeadTemporal
from .dpt import *
from .util.transform import Resize, NormalizeImage, PrepareForNet

from utils.util import compute_scale_and_shift, get_interpolate_frames
from timm.models.layers import DropPath
from torchvision.ops import DeformConv2d



# Placeholder for a Deformable Multi-Head Attention
class DeformableCrossFrameAttention(nn.Module):
    def __init__(self, dim, num_heads=8, max_points=8):
        super().__init__()
        self.num_heads = num_heads
        self.max_points = max_points
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.offset_pred = nn.Linear(dim, num_heads * max_points * 2)
        self.attn_out = nn.Linear(dim, dim)

    def forward(self, query, reference, spatial_size, n_points=None, micro_batch_size=4, frame_length=None):
        if n_points is None:
            n_points = self.max_points
        else:
            n_points = min(n_points, self.max_points)

        B, N, C = query.shape
        H, W = spatial_size
        
        ph, pw = spatial_size
        total_patches = ph * pw
        
        if n_points is None:
            if total_patches >= (384//14)**2:
                n_points = 0  # 고해상도: 1포인트
            elif total_patches >= (192//14)**2:
                n_points = 0  # 중간해상도: 2포인트
            else:
                n_points = self.max_points  # 저해상도: 최대포인트
        else:
            n_points = min(n_points, self.max_points)
        
        if frame_length is None or neighbor_frames is None:
            return self._process_all_frames(query, reference, spatial_size, n_points, micro_batch_size, B)
        else:
            return self._process_neighbor_frames(
                query, reference, spatial_size, n_points, 
                frame_length, neighbor_frames, micro_batch_size, B
            )

    def _process_all_frames(self, query, reference, spatial_size, n_points, micro_batch_size, B):
        if B <= micro_batch_size:
            return self._forward_chunk(query, reference, spatial_size, n_points)
        else:
            outputs = []
            for i in range(0, B, micro_batch_size):
                q_chunk = query[i:i+micro_batch_size]
                r_chunk = reference[i:i+micro_batch_size]
                out_chunk = self._forward_chunk(q_chunk, r_chunk, spatial_size, n_points)
                outputs.append(out_chunk)
            return torch.cat(outputs, dim=0)

    def _process_neighbor_frames(self, query, reference, spatial_size, n_points, 
                               frame_length, neighbor_frames, micro_batch_size):
        B, N, C = query.shape
        H, W = spatial_size
        num_frames = frame_length
        batch_size = B // num_frames
        
        query_3d = query.view(batch_size, num_frames, N, C)
        reference_3d = reference.view(batch_size, num_frames, N, C)
        
        outputs = []
        for t in range(num_frames):
            current_query = query_3d[:, t, :, :]
            start_idx = max(0, t - neighbor_frames)
            end_idx = min(num_frames, t + neighbor_frames + 1)
            neighbor_indices = list(range(start_idx, end_idx))
            neighbor_reference = reference_3d[:, neighbor_indices, :, :]
            num_neighbors = len(neighbor_indices)
            
            expanded_query = current_query.unsqueeze(1).repeat(1, num_neighbors, 1, 1)
            expanded_query = expanded_query.view(batch_size * num_neighbors, N, C)
            neighbor_reference = neighbor_reference.view(batch_size * num_neighbors, N, C)
            
            if batch_size * num_neighbors <= micro_batch_size:
                out = self._forward_chunk(expanded_query, neighbor_reference, spatial_size, n_points)
            else:
                chunk_out = []
                for i in range(0, batch_size * num_neighbors, micro_batch_size):
                    q_chunk = expanded_query[i:i+micro_batch_size]
                    r_chunk = neighbor_reference[i:i+micro_batch_size]
                    out_chunk = self._forward_chunk(q_chunk, r_chunk, spatial_size, n_points)
                    chunk_out.append(out_chunk)
                out = torch.cat(chunk_out, dim=0)
            
            out = out.view(batch_size, num_neighbors, N, C)
            out = out.mean(dim=1)
            outputs.append(out)
        
        return torch.stack(outputs, dim=1).view(B, N, C)

    def _forward_chunk(self, query, reference, spatial_size, n_points, chunk_size=1):
        B, N, C = query.shape
        H, W = spatial_size
        head_dim = C // self.num_heads

        # 프로젝션 레이어
        def process_projection(q, r):
            q_proj = self.q_proj(q)
            kv_proj = self.kv_proj(r)
            return q_proj, kv_proj
            
        q, kv = checkpoint(process_projection, query, reference, use_reentrant=False)
        q = q.view(B, N, self.num_heads, head_dim)
        kv = kv.view(B, N, 2, self.num_heads, head_dim)
        k, v = kv[:, :, 0], kv[:, :, 1]  # [B, N, num_heads, head_dim]

        # 오프셋 예측
        offsets = checkpoint(self.offset_pred, query, use_reentrant=False)
        offsets = offsets.view(B, N, self.num_heads, self.max_points, 2)
        offsets = offsets[:, :, :, :n_points]

        # 좌표 생성
        with torch.no_grad():
            y_coords = torch.linspace(-1, 1, H, device=query.device)
            x_coords = torch.linspace(-1, 1, W, device=query.device)
            coords = torch.stack(torch.meshgrid(y_coords, x_coords, indexing='ij'), -1)
            coords = coords.view(1, N, 1, 1, 2)
            sample_locs = coords + offsets
        
        # 그리드 샘플링
        sampled_k = self._efficient_grid_sample(
            k, sample_locs,  # k: [B, N, num_heads, head_dim]
            spatial_size, n_points,
            chunk_size=chunk_size
        )
        
        # 어텐션 계산
        q = q.unsqueeze(3)  # [B, N, heads, 1, head_dim]
        out = self._micro_batch_attention(
            q, sampled_k, self.scale,
            chunk_size=chunk_size
        )
        return self.attn_out(out)

    def _efficient_grid_sample(self, k, sample_locs, spatial_size, n_points, chunk_size=2):
        B, N, num_heads, head_dim = k.shape
        H, W = spatial_size
        
        # 최대 메모리 사용량 제한을 위한 서브 청크 설정
        head_chunk_size = max(1, 8 // n_points)  # 헤드 청크 크기
        point_chunk_size = max(1, 4)  # 포인트 청크 크기
        
        # 결과 저장용 텐서
        sampled_k = torch.zeros(
            (B, H, W, num_heads, n_points, head_dim),
            device=k.device, dtype=k.dtype
        )
        
        # 혼합 정밀도 적용
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            # 다중 청크 단위 처리
            for b in range(0, B, chunk_size):
                b_end = min(b+chunk_size, B)
                chunk_k = k[b:b_end]
                chunk_locs = sample_locs[b:b_end]
                
                for h in range(0, num_heads, head_chunk_size):
                    h_end = min(h+head_chunk_size, num_heads)
                    
                    for p in range(0, n_points, point_chunk_size):
                        p_end = min(p+point_chunk_size, n_points)
                        
                        # 현재 청크 선택
                        sub_k = chunk_k[:, :, h:h_end, :]
                        sub_locs = chunk_locs[:, :, h:h_end, p:p_end, :]
                        
                        # 그리드 샘플링 준비
                        grid = sub_locs.permute(0, 2, 3, 1, 4)  # [b, num_heads_sub, points_sub, N, 2]
                        grid = grid.reshape(-1, N, 2)  # [b*num_heads_sub*points_sub, N, 2]
                        grid = grid.view(-1, H, W, 2)  # [b*num_heads_sub*points_sub, H, W, 2]
                        
                        ref_k = sub_k.permute(0, 2, 3, 1).reshape(-1, head_dim, H, W)
                        ref_k = ref_k.reshape(-1, head_dim, H, W)  # [b*num_heads_sub, head_dim, H, W]
                        
                        # 메모리 효율적 확장
                        ref_k = ref_k.unsqueeze(1).expand(-1, p_end-p, -1, -1, -1)
                        ref_k = ref_k.reshape(-1, *ref_k.shape[2:])
                        
                        # 그리드 샘플링
                        sampled_sub = F.grid_sample(
                            ref_k, grid,
                            align_corners=True,
                            padding_mode='border',
                            mode='bilinear'
                        )
                        
                        # 결과 저장
                        sampled_sub = sampled_sub.view(
                            chunk_k.size(0), h_end-h, p_end-p, head_dim, H, W
                        )
                        sampled_sub = sampled_sub.permute(0, 4, 5, 1, 2, 3)
                        sampled_k[b:b_end, :, :, h:h_end, p:p_end, :] = sampled_sub
        
        return sampled_k.view(B, N, num_heads, n_points, head_dim)

    def _micro_batch_attention(self, q, sampled_k, scale, chunk_size=2):
        # 혼합 정밀도 적용
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            B, N, num_heads, n_points, head_dim = sampled_k.shape
            out = torch.zeros((B, N, num_heads * head_dim), 
                             device=q.device, dtype=torch.float32)
            
            # 어텐션 스코어 계산을 float32로 유지
            for b in range(0, B, chunk_size):
                b_end = min(b+chunk_size, B)
                q_chunk = q[b:b_end].float()  # float32로 변환
                k_chunk = sampled_k[b:b_end].float()  # float32로 변환
                
                attn_scores = torch.einsum('bnhpd,bnhpd->bnhp', q_chunk, k_chunk) * scale
                attn = attn_scores.softmax(-1)
                out_chunk = torch.einsum('bnhp,bnhpd->bnhd', attn, k_chunk)
                out[b:b_end] = out_chunk.reshape(b_end-b, N, -1)
            
            return out

class DPTHeadTemporalDeformable(DPTHead):
    def __init__(
        self,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
        use_clstoken=False,
        num_frames=32,
        pe='ape',
        micro_batch_size=4
    ):
        # 부모 클래스 초기화 (readout_projects, projects, resize_layers, scratch 구성)
        super().__init__(
            in_channels=in_channels,
            features=features,
            use_bn=use_bn,
            out_channels=out_channels,
            use_clstoken=use_clstoken
        )

        # 변형 어텐션 모듈 준비
        self.deform_attns = nn.ModuleList([
            DeformableCrossFrameAttention(out_channels[2], num_heads=8, max_points=8),
            DeformableCrossFrameAttention(out_channels[3], num_heads=8, max_points=8),
            DeformableCrossFrameAttention(features,    num_heads=8, max_points=8),
            DeformableCrossFrameAttention(features,    num_heads=8, max_points=8)
        ])

        self.num_frames = num_frames
        self.micro_batch_size = micro_batch_size

    def forward(self, out_features, patch_h, patch_w, frame_length, neighbor_frames=1):
        # 1) 각 레이어 특징 맵 준비
        stack = []
        for i, feat in enumerate(out_features):
            if self.use_clstoken:
                x, cls = feat
                read = cls.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat([x, read], dim=-1))
            else:
                x = feat[0]

            Bfull = x.size(0)
            T = frame_length
            B = Bfull // T
            # [B*T, N, C] -> [B*T, C, H, W]
            x = x.permute(0, 2, 1).reshape(Bfull, -1, patch_h, patch_w)
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            stack.append(x)

        layer1, layer2, layer3, layer4 = stack

        # 2) Deformable Temporal Attention 적용 함수
        def _apply_deform_attn(x, idx):
            Bfull, C, H, W = x.shape
            N = H * W
            q = x.flatten(2).permute(0, 2, 1)  # [B*T, N, C]
            r = q  # self-reference
            attn = self.deform_attns[idx]
            out = attn(
                query=q,
                reference=r,
                spatial_size=(H, W),
                n_points=None,
                micro_batch_size=self.micro_batch_size,
                frame_length=frame_length,
                neighbor_frames=neighbor_frames
            )  # [B*T, N, C]
            # 다시 [B*T, C, H, W]
            return out.permute(0, 2, 1).reshape(Bfull, C, H, W)

        # 주요 레이어에 변형 어텐션 적용
        layer3 = _apply_deform_attn(layer3, idx=0)
        layer4 = _apply_deform_attn(layer4, idx=1)

        # 3) RefineNet 연산
        layer1_rn = self.scratch.layer1_rn(layer1)
        layer2_rn = self.scratch.layer2_rn(layer2)
        layer3_rn = self.scratch.layer3_rn(layer3)
        layer4_rn = self.scratch.layer4_rn(layer4)

        path4 = self.scratch.refinenet4(layer4_rn, size=layer3_rn.shape[2:])
        path4 = _apply_deform_attn(path4, idx=2)
        path3 = self.scratch.refinenet3(path4, layer3_rn, size=layer2_rn.shape[2:])
        path3 = _apply_deform_attn(path3, idx=3)

        # 4) 최종 생성 및 마이크로 배치 처리
        Bfull = layer1_rn.shape[0]
        if Bfull <= self.micro_batch_size or Bfull % self.micro_batch_size != 0:
            path2 = self.scratch.refinenet2(path3, layer2_rn, size=layer1_rn.shape[2:])
            path1 = self.scratch.refinenet1(path2, layer1_rn)
            out = self.scratch.output_conv1(path1)
            out = F.interpolate(out, (patch_h * 14, patch_w * 14), mode='bilinear', align_corners=True)
            return self.scratch.output_conv2(out)
        else:
            chunks = []
            for i in range(0, Bfull, self.micro_batch_size):
                p2 = self.scratch.refinenet2(
                    path3[i:i+self.micro_batch_size],
                    layer2_rn[i:i+self.micro_batch_size],
                    size=layer1_rn.shape[2:]
                )
                p1 = self.scratch.refinenet1(p2, layer1_rn[i:i+self.micro_batch_size])
                o = self.scratch.output_conv1(p1)
                o = F.interpolate(o, (patch_h * 14, patch_w * 14), mode='bilinear', align_corners=True)
                chunks.append(self.scratch.output_conv2(o))
            return torch.cat(chunks, dim=0)
            

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

        # intermediate layer indices
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23]
        }
        self.encoder = encoder

        # pretrained encoder
        self.pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

        # replace DPT head with deformable temporal version
        self.head = DPTHeadTemporalDeformable(
            in_channels=self.pretrained.embed_dim,
            features=features,
            use_bn=use_bn,
            out_channels=out_channels,
            use_clstoken=use_clstoken,
            num_frames=num_frames,
            pe=pe,
            micro_batch_size=2
        )

        # diff + conv pipeline
        self.diff = diff
        self.conv = conv
        self.out_channel = out_channel
        if diff:
            if conv:
                self.diff_layers = nn.Sequential(
                    nn.Conv2d(3, out_channel//2, 3, padding=1),
                    nn.BatchNorm2d(out_channel//2),
                    nn.ReLU(True),
                    nn.Conv2d(out_channel//2, out_channel, 3, padding=1),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(True)
                )
            else:
                self.out_channel = 3
            self.mlp = nn.Conv2d(self.out_channel, self.pretrained.embed_dim, 1)

        # skip conv for prev depth
        self.skip_conv = nn.Conv2d(1, self.pretrained.embed_dim, 3, padding=1)
        self.prev_depth = None

    def forward(self, x):
        B, T, C, H, W = x.shape
        ph, pw = H // 14, W // 14

        # mixed precision
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # extract encoder features
            feats = self.pretrained.get_intermediate_layers(
                x.flatten(0,1),
                self.intermediate_layer_idx[self.encoder],
                return_class_token=True
            )

            # diff pipeline
            if self.diff:
                dt = x[:,1:] - x[:,:-1]
                dt_flat = dt.reshape(-1, C, H, W)
                if self.conv:
                    dt_flat = self.diff_layers(dt_flat)
                pooled = F.adaptive_avg_pool2d(dt_flat, (ph, pw))
                df = self.mlp(pooled).view(B, T-1, -1, ph, pw)
                df_list = []
                for i in range(T):
                    if i == 0:
                        df_list.append(df[:,0])
                    elif i == T-1:
                        df_list.append(df[:,-1])
                    else:
                        df_list.append(0.5 * (df[:,i-1] + df[:,i]))
                df_tokens = torch.stack(df_list, 1)
                df_tokens = df_tokens.permute(0,1,3,4,2).reshape(B*T, ph*pw, -1)
                updated = [(f_tok + df_tokens, cls) for (f_tok, cls) in feats]
            else:
                updated = feats

        # add skip connection from prev_depth if available
        if self.prev_depth is not None and self.prev_depth.shape == (B, T, 1, H, W):
            pd = F.interpolate(
                self.prev_depth.view(B*T,1,H,W), (ph,pw),
                mode='bilinear', align_corners=True
            )
            pd_feat = self.skip_conv(pd).view(B, T, ph*pw, -1)
            updated = [(u_tok + pd_feat.reshape(B*T, ph*pw, -1), cls)
                       for (u_tok, cls) in updated]

        # compute depth with deformable temporal head
        depth = self.head(updated, ph, pw, T)  # shape: (B*T, ph, pw)
        depth = depth.unsqueeze(1)
        depth = F.interpolate(depth, (H, W), mode='bilinear', align_corners=True)

        # clear cache and reshape
        torch.cuda.empty_cache()
        return depth.view(B, T, H, W)


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
                # 추론 시 혼합 정밀도 적용
                with torch.autocast(device_type=device, enabled=(not fp32)):
                    depth = self.forward(cur_input)  # depth shape: [1, T, H, W]

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
