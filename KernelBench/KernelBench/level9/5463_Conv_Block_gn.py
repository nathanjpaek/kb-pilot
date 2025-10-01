import torch
import torch.nn as nn
from torch.autograd.variable import *


class Conv_Block_gn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, groups, stride=1
        ):
        super(Conv_Block_gn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=
            kernel_size, stride=stride, padding=1)
        self.relu = torch.nn.LeakyReLU()
        self.bn = nn.GroupNorm(groups, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4,
        'groups': 1}]
