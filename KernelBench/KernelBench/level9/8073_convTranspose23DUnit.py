import torch
import numpy as np
from torch import nn
import torch.utils.data
import torch.nn.init as init
import torch.nn.init


class convTranspose23DUnit(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, output_padding=0, groups=1, bias=True, dilation=1, nd=2):
        super(convTranspose23DUnit, self).__init__()
        assert nd == 1 or nd == 2 or nd == 3, 'nd is not correctly specified!!!!, it should be {1,2,3}'
        if nd == 2:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels,
                kernel_size, stride=stride, padding=padding, output_padding
                =output_padding, groups=groups, bias=bias, dilation=dilation)
        elif nd == 3:
            self.conv = nn.ConvTranspose3d(in_channels, out_channels,
                kernel_size, stride=stride, padding=padding, output_padding
                =output_padding, groups=groups, bias=bias, dilation=dilation)
        else:
            self.conv = nn.ConvTranspose1d(in_channels, out_channels,
                kernel_size, stride=stride, padding=padding, output_padding
                =output_padding, groups=groups, bias=bias, dilation=dilation)
        init.xavier_uniform(self.conv.weight, gain=np.sqrt(2.0))
        init.constant(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
