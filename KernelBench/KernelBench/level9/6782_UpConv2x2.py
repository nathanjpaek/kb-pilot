import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.cuda import *


class UpConv2x2(nn.Module):

    def __init__(self, channels):
        super(UpConv2x2, self).__init__()
        self.conv = nn.Conv2d(channels, channels // 2, kernel_size=2,
            stride=1, padding=0, bias=True)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = F.pad(x, (0, 1, 0, 1))
        x = self.conv(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channels': 4}]
