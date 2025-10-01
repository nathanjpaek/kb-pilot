import functools
import torch
from torch import nn
import torch.nn.functional as F


class PixelShuffle2d(nn.Conv2d):

    def __init__(self, in_nc, out_nc, kernel_size: 'int', scale: 'int'=2,
        **kwargs):
        super().__init__(in_nc, out_nc * scale * scale, kernel_size, **kwargs)
        self.up = functools.partial(F.pixel_shuffle, upscale_factor=scale)

    def forward(self, x):
        x = super().forward(x)
        x = self.up(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_nc': 4, 'out_nc': 4, 'kernel_size': 4}]
