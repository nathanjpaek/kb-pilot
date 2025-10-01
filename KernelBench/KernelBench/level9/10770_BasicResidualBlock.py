import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
        padding=1, bias=True, normalization=None, activation='prelu'):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias)
        if normalization == 'batch':
            self.norm = nn.BatchNorm2d(out_channels)
        elif normalization == 'instance':
            self.norm = nn.InstanceNorm2d(out_channels)
        else:
            self.norm = None
        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'prelu':
            self.act = nn.PReLU()
        elif activation == 'lrelu':
            self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            self.act = None

    def forward(self, x):
        out = self.conv(x)
        if self.norm is not None:
            out = self.norm(out)
        if self.act is not None:
            out = self.act(out)
        return out


class BasicResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, bias=True,
        normalization=None, activation='prelu', downsample=None):
        super(BasicResidualBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, channels, stride=stride, bias=
            bias, normalization=normalization, activation=activation)
        self.conv2 = ConvBlock(channels, channels, bias=bias, normalization
            =normalization, activation=None)
        self.downsample = downsample
        self.prelu = nn.PReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += x if self.downsample is None else self.downsample(x)
        out = self.prelu(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'channels': 4}]
