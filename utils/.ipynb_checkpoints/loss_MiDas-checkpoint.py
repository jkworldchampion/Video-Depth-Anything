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
        sum_p = (pred_flat * mask_flat).sum(dim=1, keepdim=True)     # (B, 1)
        sum_g = (gt_flat   * mask_flat).sum(dim=1, keepdim=True)     # (B, 1)
        mu_p = sum_p / valid_counts                                  # (B, 1)
        mu_g = sum_g / valid_counts                                  # (B, 1)

        # 5) 분산/공분산: (d_i - μ_p)(d*_i - μ_g) 및 (d_i - μ_p)^2, mask 적용
        p_diff = pred_flat - mu_p    # (B, M)
        g_diff = gt_flat   - mu_g    # (B, M)
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

        # 11) batch마다 출력
        print("SSI Loss per batch:", loss.item())

        return loss
    
    
class VectorizedLossTGM(nn.Module):
    """
    Vectorized Temporal Gradient Matching Loss (TGM)

    pred, y:        B x N x 1 x H x W  또는  B x N x H x W
    masks_squeezed: B x N x H x W      (boolean mask of valid pixels)

    - 인접 프레임 동일 픽셀 위치에서의 변화량 차이가 GT 변화량 차이와 일치하도록 학습
    - GT 변화량 < 0.05인 픽셀만 계산에 포함
    """
    def __init__(self, threshold: float = 0.05, eps: float = 1e-6):
        super().__init__()
        self.threshold = threshold
        self.eps = eps

    def forward(self, pred: torch.Tensor, y: torch.Tensor, masks_squeezed: torch.Tensor) -> torch.Tensor:
        # 1) (B, N, 1, H, W) → (B, N, H, W)
        if pred.dim() == 5 and pred.shape[2] == 1:
            pred = pred.squeeze(2)
        if y.dim() == 5 and y.shape[2] == 1:
            y = y.squeeze(2)

        # 2) mask → bool
        mask = masks_squeezed.bool()  # (B, N, H, W)

        B, N, H, W = pred.shape
        if N < 2:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # 3) 인접 프레임 차이: (B, N-1, H, W)
        pred_diff = torch.abs(pred[:, 1:] - pred[:, :-1])
        gt_diff   = torch.abs(y[:, 1:]    - y[:, :-1])

        # 4) 두 프레임 모두 valid인 위치
        mask_i   = mask[:, :-1]    # (B, N-1, H, W)
        mask_ip1 = mask[:, 1:]     # (B, N-1, H, W)
        valid_pair = mask_i & mask_ip1

        # 5) GT 변화량 < threshold
        stable_gt = gt_diff < self.threshold

        # 6) 최종 valid 영역: (B, N-1, H, W)
        valid = valid_pair & stable_gt

        # 7) 절대 오차: |(pred_diff - gt_diff)|, (B, N-1, H, W)
        abs_err = torch.abs(pred_diff - gt_diff)

        # 8) i별 유효 픽셀 수: (B, N-1)
        valid_count = valid.sum(dim=(-1, -2)).float()  # sum over H, W

        # 9) i별 오차 합: (B, N-1)
        err_sum = (abs_err * valid).sum(dim=(-1, -2))

        # 10) i별 평균 오차: 0/eps 처리 포함
        error_i = err_sum / (valid_count + self.eps)  # (B, N-1)

        # 11) 샘플당 TGM: (1/(N-1)) * sum_i error_i
        sample_tgm = error_i.sum(dim=1) / float(N - 1)  # (B,)

        # 12) 배치 평균
        loss = sample_tgm.mean()

        print("TGM Loss per batch:", loss.item())
        return loss