import torch
import torch.nn as nn
import torch.nn.functional as F


class UpsamplingBilinear2d(nn.Module):

    def __init__(self, scale_factor=2.0):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=
            'bilinear', align_corners=True)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
