import torch
import torch.nn as nn


class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, norm
        =None, bias=True):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride, bias=bias)
        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels,
                track_running_stats=True)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.norm in ['BN' or 'IN']:
            out = self.norm_layer(out)
        return out


class ResidualBlock(nn.Module):

    def __init__(self, channels, norm=None, bias=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1,
            bias=bias, norm=norm)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1,
            bias=bias, norm=norm)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        input = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + input
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channels': 4}]
