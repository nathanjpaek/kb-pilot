import torch
from torch import nn


class Upsample(nn.Module):

    def __init__(self, scale_factor, mode='bilinear'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, input):
        return nn.functional.interpolate(input, scale_factor=self.
            scale_factor, mode=self.mode, align_corners=False)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'scale_factor': 1.0}]
