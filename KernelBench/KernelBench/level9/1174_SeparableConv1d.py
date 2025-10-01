import torch
import torch.nn as nn


class SeparableConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding='same', bias=False):
        super(SeparableConv1d, self).__init__()
        if stride > 1:
            padding = 0
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=
            kernel_size, groups=in_channels, bias=bias, padding=padding,
            stride=stride)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1,
            bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
