import torch
import torch.nn as nn


class ConvLayer(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size:
        'int', stride: 'int'):
        super().__init__()
        self._conv = nn.Conv2d(in_channels=in_channels, out_channels=
            out_channels, kernel_size=kernel_size, stride=stride)
        self._pad = nn.ReflectionPad2d(padding=kernel_size // 2)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = self._pad(x)
        x = self._conv(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4,
        'stride': 1}]
