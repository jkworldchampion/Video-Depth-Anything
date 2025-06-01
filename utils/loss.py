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

    def forward(self, pred, y, disparity=True):
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

        # print(pred.shape)
        # print(y.shape)

        if not disparity:
            ## 역수로 바꿔주기
            temp = torch.ones_like(y)
            y = temp / (y + self.eps)

        ## group_1에 대해서는 그냥 일반 ssi loss 적용해주면 ok

        y = self._normalize_depth(y)

        """
        print("gt ( 1st element ):", y[0])
        print("pred ( 1st element ):", pred[0])
        print("y aggregation ( 1st element ):", torch.sum(y[0], dim=-1))
        print("pred aggregation ( 1st element ):", torch.sum(pred[0], dim=-1))
        """

        loss_ssi = torch.sum(self._rho(pred, y), dim=-1) / pred.shape[-1]
        loss_ssi = loss_ssi.mean()

        return loss_ssi


class Loss_tgm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, y):
        """
        train_code에서 넘어올 때를 생각해보자,, 분명 B x N x C x H x W 사이즈로 넘어오겠지? ( 여기선 overlapping x 일듯. )
        pred, y : B x N x 1 x H x W
        """

        if pred.dim() == 5 and pred.shape[2] == 1:
            pred = pred.squeeze(2)

        if y.dim() == 5 and y.shape[2] == 1:
            y = y.squeeze(2)

        B, N, H, W = pred.shape
        pred = pred.view(B, N, H * W)
        y = y.view(B, N, H * W)

        ## 얘는 ssi loss랑 다르게, N이 좀 중요함. BxN으로 때리면 윈도우별 구분이 안가서 문제가 생김

        loss_tgm = torch.zeros(())


        for idx in range(B):
            temp = torch.zeros(())
            for d, d_next, g, g_next in zip(pred[idx][:-1], pred[idx][1:], y[idx][:-1], y[idx][1:]):
                #print(d, d_next)
                d_abs_diff = torch.abs(d - d_next)
                g_abs_diff = torch.abs(g - g_next)
                #print(d_abs_diff)
                temp += torch.sum(torch.abs(d_abs_diff - g_abs_diff))

            loss_tgm += temp / (N - 1)

        loss_tgm = loss_tgm / B

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