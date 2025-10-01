import functools
import torch
from torch import nn
from typing import Optional
import torch.nn.functional as F


class DeConv2d(nn.Conv2d):

    def __init__(self, in_nc: 'int', out_nc: 'int', kernel_size: 'int',
        scale: 'int'=2, mode: 'str'='nearest', align_corners:
        'Optional[bool]'=None, **kwargs):
        super().__init__(in_nc, out_nc, kernel_size, **kwargs)
        self.up = functools.partial(F.interpolate, scale_factor=scale, mode
            =mode, align_corners=align_corners)

    def forward(self, x):
        x = self.up(x)
        x = super().forward(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_nc': 4, 'out_nc': 4, 'kernel_size': 4}]
