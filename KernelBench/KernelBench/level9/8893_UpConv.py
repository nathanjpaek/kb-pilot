import torch
from collections import OrderedDict
import torch.nn as nn


class UpConv(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.up_conv = nn.Sequential(OrderedDict([('up', nn.Upsample(
            scale_factor=2)), ('conv', nn.Conv2d(in_channels, in_channels //
            2, kernel_size=3, padding=1))]))

    def forward(self, x):
        return self.up_conv(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4}]
