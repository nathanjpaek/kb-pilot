import math
import torch
from torch import nn
import torch.nn.functional as F


class EqualConv2d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, groups=1,
        stride=1, padding=0, bias=True, lr_mul=1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channel, in_channel //
            groups, kernel_size, kernel_size).div_(lr_mul))
        self.scale = lr_mul / math.sqrt(in_channel // groups * kernel_size ** 2
            )
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.lr_mul = lr_mul
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None

    def forward(self, input):
        self.bias * self.lr_mul if self.bias is not None else None
        out = F.conv2d(input, self.weight * self.scale, bias=self.bias,
            stride=self.stride, padding=self.padding, groups=self.groups)
        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]}, {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
            )


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channel': 4, 'out_channel': 4, 'kernel_size': 4}]
