import math
import torch
from torch import nn


class Caps_Conv(nn.Module):

    def __init__(self, in_C, in_D, out_C, out_D, kernel_size, stride=1,
        padding=0, dilation=1, bias=False):
        super(Caps_Conv, self).__init__()
        self.in_C = in_C
        self.in_D = in_D
        self.out_C = out_C
        self.out_D = out_D
        self.conv_D = nn.Conv2d(in_C * in_D, in_C * out_D, 1, groups=in_C,
            bias=False)
        self.conv_C = nn.Conv2d(in_C, out_C, kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=bias)
        m = self.conv_D.kernel_size[0] * self.conv_D.kernel_size[1
            ] * self.conv_D.out_channels
        self.conv_D.weight.data.normal_(0, math.sqrt(2.0 / m))
        n = self.conv_C.kernel_size[0] * self.conv_C.kernel_size[1
            ] * self.conv_C.out_channels
        self.conv_C.weight.data.normal_(0, math.sqrt(2.0 / n))
        if bias:
            self.conv_C.bias.data.zero_()

    def forward(self, x):
        x = self.conv_D(x)
        x = x.view(x.shape[0], self.in_C, self.out_D, x.shape[2], x.shape[3])
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(-1, self.in_C, x.shape[3], x.shape[4])
        x = self.conv_C(x)
        x = x.view(-1, self.out_D, self.out_C, x.shape[2], x.shape[3])
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(-1, self.out_C * self.out_D, x.shape[3], x.shape[4])
        return x


def get_inputs():
    return [torch.rand([4, 16, 64, 64])]


def get_init_inputs():
    return [[], {'in_C': 4, 'in_D': 4, 'out_C': 4, 'out_D': 4,
        'kernel_size': 4}]
