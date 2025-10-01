import torch
import torch.nn as nn
import torch.nn.functional as F


class FRN(nn.Module):

    def __init__(self, num_features, eps=1e-05):
        super(FRN, self).__init__()
        self.tau = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.eps = eps

    def forward(self, x):
        x = x * torch.rsqrt(torch.mean(x ** 2, dim=[2, 3], keepdim=True) +
            self.eps)
        return torch.max(self.gamma * x + self.beta, self.tau)


class ResBlk(nn.Module):
    """Preactivation residual block with filter response norm."""

    def __init__(self, dim_in, dim_out, style_dim=64, downsample=False):
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.downsample = downsample
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.norm1 = FRN(dim_in)
        self.norm2 = FRN(dim_in)
        if self.downsample:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 4, 2, 1)

    def _shortcut(self, x):
        if self.downsample:
            x = self.conv1x1(x)
        return x

    def _residual(self, x):
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.norm2(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._residual(x) + self._shortcut(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_in': 4, 'dim_out': 4}]
