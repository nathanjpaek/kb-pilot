import torch
from torch import nn
import torch.utils.data
import torch


class Upsample(nn.Module):

    def __init__(self, scale):
        super(Upsample, self).__init__()
        self.up = nn.Upsample(scale_factor=scale, mode='bicubic',
            align_corners=True)

    def forward(self, x):
        return self.up(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'scale': 1.0}]
