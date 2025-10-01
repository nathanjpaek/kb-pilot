import torch
import torch.nn as nn
import torch.nn.functional as F


class Interpolate(nn.Module):

    def __init__(self, scale_factor, mode='bilinear', align_corners=True):
        super(Interpolate, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode,
            align_corners=self.align_corners)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'scale_factor': 1.0}]
