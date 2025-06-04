# This version is different from the Depth Anything V2 loss function.
# It is same as the MiDaS loss function. It uses the MSE(not MAE) for the loss calculation.

import torch
import torch.nn as nn

class MaskedScaleShiftInvariantLoss(nn.Module):
    """
    Scale- & Shift-Invariant MSE Loss with masking (video sequence 전체를 alignment 단위로 사용)
    - pred: 예측된 disparity, shape = (B, N, 1, H, W)
    - gt:    ground-truth disparity, shape = (B, N, 1, H, W)
    - mask:  valid 마스크, shape = (B, N, H, W), bool 또는 0/1 tensor
    """
    def __init__(self, eps: float = 1e-6):
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
        denom    = torch.sum((p_diff * p_diff) * mask_flat,    dim=1, keepdim=True) # (B, 1)

        # 6) s, t 계산
        s = numerator / (denom + self.eps)  # (B, 1)
        t = mu_g - s * mu_p                 # (B, 1)

        # 7) aligned prediction: ŷ = s * p + t
        aligned_pred = s * pred_flat + t    # (B, M)

        # 8) residual = (ŷ - gt)^2, mask 적용
        residual = (aligned_pred - gt_flat) ** 2  # (B, M)
        masked_residual = residual * mask_flat    # invalid 위치는 0

        # 9) per-sample loss: 1/(2 * valid_counts) * sum(masked_residual)
        loss_per_sample = masked_residual.sum(dim=1, keepdim=True) / (2.0 * valid_counts)  # (B, 1)

        # 10) 배치 평균 반환
        loss = loss_per_sample.mean()  # scalar
        return loss

