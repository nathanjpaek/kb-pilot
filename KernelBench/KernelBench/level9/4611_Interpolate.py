import torch
import torch.nn as nn
import torch.nn.functional as F


class Interpolate(nn.Module):

    def __init__(self):
        super(Interpolate, self).__init__()

    def forward(self, x):
        x = F.interpolate(x, scale_factor=8, mode='nearest', align_corners=None
            )
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
