import torch
from torch import nn
from typing import *
from typing import Optional


class AdaptiveConcatPool2d(nn.Module):
    """Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`"""

    def __init__(self, sz: 'Optional[int]'=None):
        """Output will be 2*sz or 2 if sz is None"""
        super().__init__()
        sz = sz or 1
        self.ap, self.mp = nn.AdaptiveAvgPool2d(sz), nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
