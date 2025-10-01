import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.utils.data
import torch.nn.init as init
import torch.nn.init


class conv23DUnit(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, groups=1, bias=True, dilation=1, nd=2):
        super(conv23DUnit, self).__init__()
        assert nd == 1 or nd == 2 or nd == 3, 'nd is not correctly specified!!!!, it should be {1,2,3}'
        if nd == 2:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, groups=groups, bias=bias,
                dilation=dilation)
        elif nd == 3:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, groups=groups, bias=bias,
                dilation=dilation)
        else:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, groups=groups, bias=bias,
                dilation=dilation)
        init.xavier_uniform(self.conv.weight, gain=np.sqrt(2.0))
        init.constant(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


class unetConvUnit(nn.Module):

    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu,
        nd=2):
        super(unetConvUnit, self).__init__()
        self.conv = conv23DUnit(in_size, out_size, kernel_size=3, stride=1,
            padding=1, nd=nd)
        self.conv2 = conv23DUnit(out_size, out_size, kernel_size=3, stride=
            1, padding=1, nd=nd)
        self.activation = activation

    def forward(self, x):
        out = self.activation(self.conv(x))
        out = self.activation(self.conv2(out))
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_size': 4, 'out_size': 4}]
