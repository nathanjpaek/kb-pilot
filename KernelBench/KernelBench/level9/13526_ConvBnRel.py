import torch
from torch.autograd.gradcheck import *
import torch.nn as nn
import torch.nn


class ConvBnRel(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        active_unit='relu', same_padding=False, bn=False, reverse=False,
        bias=False):
        super(ConvBnRel, self).__init__()
        padding = int((kernel_size - 1) // 2) if same_padding else 0
        if not reverse:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                stride, padding=padding, bias=bias)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels,
                kernel_size, stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.0001, momentum=0,
            affine=True) if bn else None
        if active_unit == 'relu':
            self.active_unit = nn.ReLU(inplace=True)
        elif active_unit == 'elu':
            self.active_unit = nn.ELU(inplace=True)
        else:
            self.active_unit = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.active_unit is not None:
            x = self.active_unit(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
