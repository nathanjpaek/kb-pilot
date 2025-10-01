import torch
import numpy as np
import torch.nn as nn


class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(nn.Module):
    """
    ResidualBlock

    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(num_channels, num_channels, kernel_size=3,
            stride=1)
        self.instance1 = nn.InstanceNorm2d(num_channels, affine=True)
        self.conv2 = ConvLayer(num_channels, num_channels, kernel_size=3,
            stride=1)
        self.instance2 = nn.InstanceNorm2d(num_channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.relu(self.instance1(self.conv1(x)))
        x = self.instance2(self.conv2(x))
        x += residual
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_channels': 4}]
