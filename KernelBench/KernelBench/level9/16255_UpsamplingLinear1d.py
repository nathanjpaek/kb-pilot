import torch
import torch.nn.functional as F
import torch.nn as nn


class UpsamplingLinear1d(nn.Module):

    def __init__(self, scale_factor=2.0):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=
            'linear', align_corners=True)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {}]
