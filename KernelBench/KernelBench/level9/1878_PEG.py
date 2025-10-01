import torch
import torch.nn as nn


class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PEG(nn.Module):

    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.proj = Residual(nn.Conv2d(dim, dim, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=dim, stride=1))

    def forward(self, x):
        return self.proj(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
