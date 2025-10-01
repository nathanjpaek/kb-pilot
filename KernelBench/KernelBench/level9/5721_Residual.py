import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


class ConvBnRelu(nn.Module):
    """
        A block of convolution, relu, batchnorm
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
        padding=0):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride, padding)
        self.bn = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvTripleBlock(nn.Module):
    """
        A block of 3 ConvBnRelu blocks.
        This triple block makes up a residual block as described in the paper
        Resolution h x w does not change across this block
    """

    def __init__(self, in_channels, out_channels):
        super(ConvTripleBlock, self).__init__()
        out_channels_half = out_channels // 2
        self.convblock1 = ConvBnRelu(in_channels, out_channels_half)
        self.convblock2 = ConvBnRelu(out_channels_half, out_channels_half,
            kernel_size=3, stride=1, padding=1)
        self.convblock3 = ConvBnRelu(out_channels_half, out_channels)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        return x


class SkipLayer(nn.Module):
    """
        The skip connections are necessary for transferring global and local context
        Resolution h x w does not change across this block
    """

    def __init__(self, in_channels, out_channels):
        super(SkipLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels != out_channels:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            x = self.conv(x)
        return x


class Residual(nn.Module):
    """
        The highly used Residual block
        Resolution h x w does not change across this block
    """

    def __init__(self, in_channels, out_channels):
        super(Residual, self).__init__()
        self.convblock = ConvTripleBlock(in_channels, out_channels)
        self.skip = SkipLayer(in_channels, out_channels)

    def forward(self, x):
        y = self.convblock(x)
        z = self.skip(x)
        out = y + z
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
