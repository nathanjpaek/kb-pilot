import math
import torch
from torch import nn


class Conv2D(nn.Module):

    def __init__(self, in_channels, kernel_size, last):
        super().__init__()
        if last:
            out_channels = 1
        else:
            out_channels = 5
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=
            kernel_size, padding=int(math.floor(kernel_size / 2)))

    def forward(self, x):
        x = self.conv2d(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'kernel_size': 4, 'last': 4}]
