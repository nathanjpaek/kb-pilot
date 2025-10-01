import torch
import torch.nn as nn


class UpsampleConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        scale_factor):
        super(UpsampleConvLayer, self).__init__()
        self._scale_factor = scale_factor
        self._reflection_pad = nn.ReflectionPad2d(kernel_size // 2)
        self._conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x = nn.functional.interpolate(x, mode='nearest', scale_factor=self.
            _scale_factor)
        x = self._reflection_pad(x)
        x = self._conv(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4,
        'stride': 1, 'scale_factor': 1.0}]
