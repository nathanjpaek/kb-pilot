import torch
import torch.nn as nn
from math import sqrt as sqrt
from itertools import product as product
from torch.nn import init as init


class FirstOctaveConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5,
        stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(FirstOctaveConv, self).__init__()
        self.stride = stride
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.h2l = torch.nn.Conv2d(in_channels, int(alpha * out_channels),
            kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(in_channels, out_channels - int(alpha *
            out_channels), kernel_size, 1, padding, dilation, groups, bias)

    def forward(self, x):
        if self.stride == 2:
            x = self.h2g_pool(x)
        X_h2l = self.h2g_pool(x)
        X_h = x
        X_h = self.h2h(X_h)
        X_l = self.h2l(X_h2l)
        return X_h, X_l


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
