import torch
import torch.nn as nn

class Loss_ssi(nn.Module):
    # DA와는 다르게, 들어오는 차원이 B x N x 1 x H x W 임 !!

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def _normalize_depth(self, depth_tensor):
        # min-max normalization
        # per image
        B_N, _ = depth_tensor.shape
        normalized = torch.empty_like(depth_tensor)
        for i in range(B_N):
            img = depth_tensor[i]
            min_val = img.min()
            max_val = img.max()
            normalized[i] = (img - min_val) / (max_val - min_val + self.eps)
        return normalized

    def _d_hat(self, d):
        # 각 배치별 계산을 위해 dim 설정
        # 여기서는 (BxN) x (HxW) 로 2차원이라고 가정
        median, _ = torch.median(d, dim=-1)  # 이러면 나오는 결과 : (BxN) x1
        # print(median)
        t = median.unsqueeze(-1)
        t = t.expand(-1, d.shape[-1])

        # t = torch.matmul(median.unsqueeze(0), t)
        s = torch.sum(torch.abs(d - t), dim=-1) / (d.shape[-1]) + self.eps
        # s : (BxN)x1 사이즈

        s = s.unsqueeze(-1).expand(-1, d.shape[-1])
        # print("d : ", d)
        # print("t : ", t)
        # print("s : ", s)
        # print("분자 : ", torch.sum(d - t, dim=-1))

        return (d - t) / s

    def _rho(self, pred, y):
        # print("d_hat pred: ", self._d_hat(pred))
        # print("d_hat y: ", self._d_hat(y))
        return torch.abs(self._d_hat(pred) - self._d_hat(y))

    def forward(self, pred, y, masks_squeezed,disparity=True):
        """
        :param pred: Prediction per pixel. size : BxHxW
        :param y: Ground truth. size : BxHxW
        """

        if pred.dim() == 5 and pred.shape[2] == 1:  # depth map이라 1차원 앞에 있다면? - channel
            pred = pred.squeeze(2)

        if y.dim() == 5 and y.shape[2] == 1:
            y = y.squeeze(2)

        B, N, H, W = pred.shape
        pred = pred.view(-1, H * W)
        y = y.view(-1, H * W)
        mask_flat = masks_squeezed.view(-1, H * W).bool()

        # print(pred.shape)
        # print(y.shape)

        if not disparity:
            ## 역수로 바꿔주기
            temp = torch.ones_like(y)
            y = temp / (y + self.eps)

        ## group_1에 대해서는 그냥 일반 ssi loss 적용해주면 ok

        #y = self._normalize_depth(y)

        """
        print("gt ( 1st element ):", y[0])
        print("pred ( 1st element ):", pred[0])
        print("y aggregation ( 1st element ):", torch.sum(y[0], dim=-1))
        print("pred aggregation ( 1st element ):", torch.sum(pred[0], dim=-1))
        """

        
        rho = self._rho(pred, y)  
        rho[~mask_flat] = 0    

        valid_counts = mask_flat.sum(dim=-1).clamp_min(1.0)
        #print("Valid counts per image:", valid_counts)
        loss_per_image = rho.sum(dim=-1) / valid_counts
        loss_ssi = loss_per_image.mean() 
        
        print("SSI Loss per batch:", loss_ssi)

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