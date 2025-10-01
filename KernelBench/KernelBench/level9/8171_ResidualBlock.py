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


class ResidualBlock(nn.Module):

    def __init__(self, channels: 'int', kernel_size: 'int'=3):
        super(ResidualBlock, self).__init__()
        self._conv1 = ConvLayer(in_channels=channels, out_channels=channels,
            kernel_size=kernel_size, stride=1)
        self._in1 = nn.InstanceNorm2d(num_features=channels, affine=True)
        self._relu = nn.ReLU()
        self._conv2 = ConvLayer(in_channels=channels, out_channels=channels,
            kernel_size=kernel_size, stride=1)
        self._in2 = nn.InstanceNorm2d(num_features=channels, affine=True)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        residual = x
        out = self._relu(self._in1(self._conv1(x)))
        out = self._in2(self._conv2(out))
        out = out + residual
        out = self._relu(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channels': 4}]
