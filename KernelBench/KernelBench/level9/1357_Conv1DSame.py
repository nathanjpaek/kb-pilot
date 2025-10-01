import math
import torch
import torch.nn as nn


class Conv1DSame(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, groups=1,
        bias=True, stride=1):
        super(Conv1DSame, self).__init__()
        p = (kernel_size - 1) / 2
        self.padding = nn.ConstantPad1d((math.floor(p), math.ceil(p)), 0.0)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
            groups=groups, bias=bias, stride=stride)

    def forward(self, x):
        x = self.padding(x)
        out = self.conv(x)
        return out


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
