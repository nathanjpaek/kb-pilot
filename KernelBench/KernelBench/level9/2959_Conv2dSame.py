import math
import torch
import torch.utils.data
import torch
import torch.nn as nn


class Conv2dSame(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        dilation=1):
        super(Conv2dSame, self).__init__()
        self.F = kernel_size
        self.S = stride
        self.D = dilation
        self.layer = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride, dilation=dilation)

    def forward(self, x_in):
        _N, _C, H, W = x_in.shape
        H2 = math.ceil(H / self.S)
        W2 = math.ceil(W / self.S)
        Pr = (H2 - 1) * self.S + (self.F - 1) * self.D + 1 - H
        Pc = (W2 - 1) * self.S + (self.F - 1) * self.D + 1 - W
        x_pad = nn.ZeroPad2d((Pr // 2, Pr - Pr // 2, Pc // 2, Pc - Pc // 2))(
            x_in)
        x_relu = nn.ReLU()(x_pad)
        x_out = self.layer(x_relu)
        return x_out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
