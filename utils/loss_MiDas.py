# This version is different from the Depth Anything V2 loss function.
# It is same as the MiDaS loss function. It uses the MSE(not MAE) for the loss calculation.

import torch
import torch.nn as nn

class Loss_ssi(nn.Module):
    # DA와는 다르게, 들어오는 차원이 B x N x 1 x H x W 임 !!
    # mask 차원 : [B, T, H, W]
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def _d_hat(self, d, ref, mask):  # 정렬해야할 ref 추가
        ## 자 지금 d input : B , N , H , W 임 .
        ## -> 일단 끝에 2차원에다가 각 마스크를 씌워줘야함
        B, N, H, W = mask.shape

        # 1) 각 프레임별로 flatten: (B*N, H*W)
        flat_d    = d.view(B * N, H * W)
        flat_ref = ref.view(B * N, H * W)  # ref도 flatten
        flat_mask = mask.view(B * N, H * W)
        
        # 2) valid pixel 개수 per frame: shape = (B*N, 1)
        valid_counts = flat_mask.sum(dim=1, keepdim=True).clamp_min(1.0)  # valid 픽셀 수 (B*N, 1)

        # 3) masked means μ_d, μ_ref per frame
        sum_d   = (flat_d   * flat_mask).sum(dim=1, keepdim=True)  # (B*N, 1)
        sum_ref = (flat_ref * flat_mask).sum(dim=1, keepdim=True)  # (B*N, 1)
        mu_d    = sum_d   / valid_counts  # (B*N, 1)
        mu_ref  = sum_ref / valid_counts  # (B*N, 1)

        # 4) centered differences
        d_diff   = flat_d   - mu_d    # (B*N, M)
        ref_diff = flat_ref - mu_ref  # (B*N, M)

        # 5) numerator, denom for least squares per frame
        numerator = torch.sum((d_diff * ref_diff) * flat_mask, dim=1, keepdim=True)  # (B*N, 1)
        denom     = torch.sum((d_diff * d_diff)   * flat_mask, dim=1, keepdim=True)  # (B*N, 1)

        # 6) scale s and shift t per frame
        s = numerator / (denom + self.eps)      # (B*N, 1)
        t = mu_ref - s * mu_d                   # (B*N, 1)

        # 7) aligned prediction: d_hat_flat = s * d_flat + t
        d_hat_flat = s * flat_d + t             # (B*N, M)

        # 8) reshape back to (B, N, H, W)
        d_hat = d_hat_flat.view(B, N, H, W)     # (B, N, H, W)
        return d_hat

    def _rho(self, pred, y, mask):
        # pred를 y에 맞춰서 정렬
        aligned_pred = self._d_hat(pred, y, mask)  # (B, N, H, W)
        return (aligned_pred - y) ** 2

    def forward(self, pred, y, masks_squeezed):
        # mask 차원 : [B, T, H, W]
        if pred.dim() == 5 and pred.shape[2] == 1:
            pred = pred.squeeze(2)

        if y.dim() == 5 and y.shape[2] == 1:
            y = y.squeeze(2)
            
        masks_squeezed = masks_squeezed.bool()  

        rho = self._rho(pred, y, masks_squeezed)        ## 리턴차원 : B T H W
        rho[~masks_squeezed] = 0

        # valid_counts = masks_squeezed.sum(dim=-1).clamp_min(1.0)
        # loss_per_image = rho.sum(dim=-1) / valid_counts
        # loss_ssi = loss_per_image.mean()
        
        valid_counts = masks_squeezed.sum(dim=(2, 3)).clamp_min(1.0)  # shape = (B, N)
        sum_rho = rho.sum(dim=(2, 3))  # shape = (B, N)
        loss_per_frame = sum_rho / valid_counts  # shape = (B, N)

        loss_ssi = loss_per_frame.mean()  # scalar
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

                # 정적 영역 >> GT 차이가 0에 가까운 픽셀만 TGM에 포함!! |g_next - g| < 0.05
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

class Loss_ssi_mse(nn.Module):
    """
    Scale- & Shift-Invariant MSE Loss with masking (video sequence 전체를 alignment 단위로 사용)
    - pred: 예측된 disparity, shape = (B, N, 1, H, W)
    - gt:    ground-truth disparity, shape = (B, N, 1, H, W)
    - mask:  valid 마스크, shape = (B, N, H, W), bool 또는 0/1 tensor
    """
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 예측값, torch.Tensor of shape (B, N, 1, H, W)
            gt:   GT값,     torch.Tensor of shape (B, N, 1, H, W)
            mask: valid 마스크, torch.Tensor of shape (B, N, H, W) (bool or 0/1)

        Returns:
            loss: 배치 평균 스칼라 텐서
        """
        # 1) shape 정리: 채널 차원(1) 제거 → pred/gt: (B, N, H, W)
        if pred.dim() == 5 and pred.shape[2] == 1:
            pred = pred.squeeze(2)
        if gt.dim() == 5 and gt.shape[2] == 1:
            gt = gt.squeeze(2)

        # mask를 float로 변환 (True→1.0, False→0.0)
        mask = mask.float()  # shape = (B, N, H, W)

        B, N, H, W = pred.shape
        M = N * H * W  # 전체 픽셀 수 (시퀀스 포함)

        # 2) Flatten: (B, N, H, W) → (B, M)
        pred_flat = pred.view(B, -1)   # (B, M)
        gt_flat   = gt.view(B, -1)     # (B, M)
        mask_flat = mask.view(B, -1)   # (B, M)

        # 3) valid 픽셀 개수 (per sample)
        valid_counts = mask_flat.sum(dim=1, keepdim=True)            # (B, 1)
        valid_counts = valid_counts.clamp_min(1.0)                   # 0 피하는 용

        # 4) masked 평균 (μ_p, μ_g)
        #    pred_flat * mask_flat: invalid 부분은 0으로 만들어 합계 계산
        sum_p = (pred_flat * mask_flat).sum(dim=1, keepdim=True)     # (B, 1)
        sum_g = (gt_flat   * mask_flat).sum(dim=1, keepdim=True)     # (B, 1)
        mu_p = sum_p / valid_counts                                   # (B, 1)
        mu_g = sum_g / valid_counts                                   # (B, 1)

        # 5) 분산/공분산: (d_i - μ_p)(d*_i - μ_g) 및 (d_i - μ_p)^2, mask 적용
        p_diff = pred_flat - mu_p    # (B, M)
        g_diff = gt_flat   - mu_g    # (B, M)
        # mask_flat이 1인 위치에서만 합산
        numerator = torch.sum((p_diff * g_diff) * mask_flat, dim=1, keepdim=True)   # (B, 1)
        denom    = torch.sum((p_diff * p_diff) * mask_flat, dim=1, keepdim=True) # (B, 1)

        # 6) s, t 계산
        s = numerator / (denom + self.eps)  # (B, 1)
        t = mu_g - s * mu_p                 # (B, 1)

        # 7) aligned prediction: ŷ = s * p + t
        aligned_pred = s * pred_flat + t    # (B, M)

        # 8) residual = (ŷ - gt)^2, mask 적용
        # residual = (aligned_pred - gt_flat) ** 2  # (B, M)
        residual = torch.abs(aligned_pred - gt_flat)
        masked_residual = residual * mask_flat    # invalid 위치는 0

        # 9) per-sample loss: 1/(2 * valid_counts) * sum(masked_residual)
        loss_per_sample = masked_residual.sum(dim=1, keepdim=True) / (2.0 * valid_counts)  # (B, 1)

        # 10) 배치 평균 반환
        loss = loss_per_sample.mean()  # scalar
        return loss

