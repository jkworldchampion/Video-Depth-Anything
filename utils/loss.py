import torch
import torch.nn as nn

import torch
import torch.nn as nn

class Loss_ssi(nn.Module):
    # DA와는 다르게, 들어오는 차원이 B x N x 1 x H x W 임 !!
    # mask 차원 : [B, T, H, W]
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def _d_hat(self, d, mask):
        ## 자 지금 d input : B , N , H , W 임 .
        ## -> 일단 끝에 2차원에다가 각 마스크를 씌워줘야함
        B, N, H, W = mask.shape

        flat_d    = d.view(B * N, H * W)
        flat_mask = mask.view(B * N, H * W)

        medians = torch.zeros((B * N,), device=d.device, dtype=d.dtype)
        scales  = torch.zeros((B * N,), device=d.device, dtype=d.dtype)

        for idx in range(B * N):
            vec       = flat_d[idx] # H*W 짜리 1d 텐서
            vec_mask  = flat_mask[idx]

            valid_vals = vec[vec_mask]   # 이러면 1차원 텐서 나옴

            if valid_vals.numel() == 0:
                # 만약 valid 없을때 처리
                med  = vec.new_tensor(0.0)
                sc   = vec.new_tensor(self.eps)
            else:
                # t(d)
                med = torch.median(valid_vals)  # -> 지금 valid_vals는 1d 텐서 여기서 median
                # s(d)
                abs_diff = torch.abs(valid_vals - med)
                sc = abs_diff.mean() + self.eps  

            medians[idx] = med  # (B*N) 차원 텐서
            scales[idx]  = sc

        median_matrix = medians.unsqueeze(1).expand(-1, H * W)  # (B*N, H*W)로 복원
        scale_matrix  = scales.unsqueeze(1).expand(-1, H * W)

        d_hat_flat = (flat_d - median_matrix) / scale_matrix

        d_hat = d_hat_flat.view(B, N, H, W) ## 이러고 나면, scale과 median이 valid 기준으로 만들어짐 ! !

        return d_hat

    def _rho(self, pred, y, mask):
        return torch.abs(self._d_hat(pred, mask) - self._d_hat(y, mask))

    def forward(self, pred, y, masks_squeezed):
        # mask 차원 : [B, T, H, W]
        if pred.dim() == 5 and pred.shape[2] == 1:
            pred = pred.squeeze(2)

        if y.dim() == 5 and y.shape[2] == 1:
            y = y.squeeze(2)
            
        masks_squeezed = masks_squeezed.bool()  

        rho = self._rho(pred, y, masks_squeezed)        ## 리턴차원 : B T H W
        rho[~masks_squeezed] = 0

        valid_counts = masks_squeezed.sum(dim=-1).clamp_min(1.0)
        loss_per_image = rho.sum(dim=-1) / valid_counts
        loss_ssi = loss_per_image.mean()

        print("SSI Loss per batch:", loss_ssi.item())

        return loss_ssi

class Loss_tgm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, y, masks_squeezed):
        """
        pred, y           : B x N x 1 x H x W  (disparity predictions and GT)
        masks_squeezed    : B x N x H x W      (boolean mask of valid pixels)
        """

        if pred.dim() == 5 and pred.shape[2] == 1:
            pred = pred.squeeze(2)      
        if y.dim() == 5 and y.shape[2] == 1:
            y = y.squeeze(2)           

        B, N, H, W = pred.shape
        pred_flat = pred.view(B, N, H * W)    
        y_flat    = y.view(B, N, H * W)       
        masks_flat = masks_squeezed.view(B, N, H * W).bool()  # → B x N x (H*W), bool

        loss_tgm = torch.zeros((), device=pred.device)

        for b in range(B):
            temp_b = torch.zeros((), device=pred.device)

            for i in range(N - 1):
                d_i      = pred_flat[b, i]     
                d_next   = pred_flat[b, i + 1]   
                g_i      = y_flat[b, i]  
                g_next   = y_flat[b, i + 1]      

                mask_i      = masks_flat[b, i]      
                mask_next   = masks_flat[b, i + 1]   

                # 두 프레임에 모두 valid한 픽셀만 고려
                valid = mask_i & mask_next        
                num_valid = valid.sum().item()

                if num_valid == 0:
                    continue

                d_diff = torch.abs(d_next - d_i)             
                g_diff = torch.abs(g_next - g_i)             

                # 정적 영역: GT 차이가 0에 가까운 픽셀만 TGM에 포함!! |g_next - g| < 0.05
                static_region = (torch.abs(g_next - g_i) < 0.05) & valid  
                num_static = static_region.sum().item()

                if num_static == 0:
                    continue

                diff = torch.abs(d_diff - g_diff)   

                diff_static = diff[static_region]  
                sum_diff = diff_static.sum()               

                tgm_pair = sum_diff / float(num_static)

                temp_b += tgm_pair

            loss_tgm += temp_b / float(N - 1)


        loss_tgm = loss_tgm / float(B)

        print("TGM Loss per batch:", loss_tgm.item())
        return loss_tgm
"""

# ─────────── dummy  ───────────
B, N, C, H, W = 2, 3, 1, 2, 3

pred = torch.rand(B, N, C, H, W)
y = torch.rand(B, N, C, H, W)

#print(pred.shape)
#print(pred)

criterion = Loss_tgm()
loss_value = criterion(pred, y)
print("TGM Loss =", loss_value.item())

"""