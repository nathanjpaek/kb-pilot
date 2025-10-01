import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision.transforms import functional as F
from torch.nn import functional as F


def box_filter(x, k):
    if k % 2 == 0:
        k = k + 1
    p = k // 2
    xp = F.pad(x, (1 + p, p, 1 + p, p), mode='constant', value=0)
    x_cumsum = xp.cumsum(dim=2)
    y = x_cumsum[:, :, k:, :] - x_cumsum[:, :, :-k, :]
    y_cumsum = y.cumsum(dim=3)
    z = y_cumsum[:, :, :, k:] - y_cumsum[:, :, :, :-k]
    return z


class ReceptiveFieldNorm(nn.Module):

    def __init__(self, min_scale=1 / 20, max_scale=1 / 5, eps=0.001, rate=4,
        subsample=3, coarse_to_fine=True):
        super(ReceptiveFieldNorm, self).__init__()
        self.eps = eps
        self.subsample = subsample
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.rate = rate
        self.coarse_to_fine = coarse_to_fine

    def forward(self, x, win_size=None):
        x = x.contiguous()
        _N, _C, H, W = x.size()
        if self.coarse_to_fine:
            scale = self.max_scale
        else:
            scale = self.min_scale
        it = 1
        while True:
            win_size = int(max(H, W) * scale)
            if win_size < 3 and it == 1:
                mean = x.mean(dim=(1, 2, 3), keepdim=True)
                std = x.std(dim=(1, 2, 3), keepdim=True) + self.eps
                x = x / std - mean / std
            else:
                if self.subsample > 1 and min(H, W
                    ) > self.subsample * 10 and win_size > self.subsample * 5:
                    xs = F.interpolate(x, scale_factor=1 / self.subsample,
                        mode='bilinear')
                    win_size = win_size // self.subsample
                else:
                    xs = x
                    win_size = win_size
                _, _, h, w = xs.shape
                ones = torch.ones(1, 1, h, w, dtype=x.dtype, device=x.device)
                M = box_filter(ones, win_size)
                x_mean = box_filter(xs, win_size).mean(dim=1, keepdim=True) / M
                x2_mean = box_filter(xs ** 2, win_size).mean(dim=1, keepdim
                    =True) / M
                var = torch.clamp(x2_mean - x_mean ** 2, min=0.0) + self.eps
                std = var.sqrt()
                a = 1 / std
                b = -x_mean / std
                mean_a = box_filter(a, win_size) / M
                mean_b = box_filter(b, win_size) / M
                if self.subsample > 1:
                    mean_a = F.interpolate(mean_a, size=(H, W), mode='bilinear'
                        )
                    mean_b = F.interpolate(mean_b, size=(H, W), mode='bilinear'
                        )
                x = mean_a * x + mean_b
            it += 1
            if self.coarse_to_fine:
                scale /= self.rate
                if scale < self.min_scale:
                    break
            else:
                scale *= self.rate
                if scale > self.max_scale:
                    break
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
