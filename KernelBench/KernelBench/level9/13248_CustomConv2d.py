import torch
import torch.nn as nn


class CustomConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=None, bias=True, residual_init=True):
        super(CustomConv2d, self).__init__()
        self.residual_init = residual_init
        if padding is None:
            padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias)

    def forward(self, input):
        return self.conv(input)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
