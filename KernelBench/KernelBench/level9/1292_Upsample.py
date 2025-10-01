import torch
import torch.nn as nn


class Upsample(nn.Module):

    def __init__(self, factor):
        super(Upsample, self).__init__()
        self.factor = factor

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.factor, mode=
            'bilinear', align_corners=False)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'factor': 4}]
