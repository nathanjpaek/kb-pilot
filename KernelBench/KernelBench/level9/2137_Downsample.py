import torch
from torch import Tensor
from torch import nn


class Downsample(nn.Module):
    """Downsample transition stage"""

    def __init__(self, c1, c2):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, 3, 2, 1)

    def forward(self, x: 'Tensor') ->Tensor:
        x = x.permute(0, 3, 1, 2)
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'c1': 4, 'c2': 4}]
