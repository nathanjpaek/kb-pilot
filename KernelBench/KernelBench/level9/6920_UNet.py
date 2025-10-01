import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.cuda import *


def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
        padding=1, bias=True)


def maxpool2x2():
    return nn.MaxPool2d(kernel_size=2, stride=2, padding=0)


def concat(xh, xv):
    return torch.cat([xh, xv], dim=1)


class UpConv2x2(nn.Module):

    def __init__(self, channels):
        super(UpConv2x2, self).__init__()
        self.conv = nn.Conv2d(channels, channels // 2, kernel_size=2,
            stride=1, padding=0, bias=True)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = F.pad(x, (0, 1, 0, 1))
        x = self.conv(x)
        return x


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels: number of channels in input (1st) feature map
            out_channels: number of channels in output feature maps
        """
        super(ConvBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.conv3 = conv3x3(out_channels, out_channels)
        self.norm = nn.BatchNorm2d(out_channels, track_running_stats=False)

    def forward(self, x):
        x = F.relu(self.norm(self.conv1(x)))
        x = F.relu(self.norm(self.conv2(x)))
        x = F.relu(self.norm(self.conv3(x)))
        return x


class DownConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels: number of channels in input (1st) feature map
            out_channels: number of channels in output feature maps
        """
        super(DownConvBlock, self).__init__()
        self.maxpool = maxpool2x2()
        self.conv1 = conv3x3(in_channels, out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.conv3 = conv3x3(out_channels, out_channels)
        self.norm = nn.BatchNorm2d(out_channels, track_running_stats=False)

    def forward(self, x):
        x = self.maxpool(x)
        x = F.relu(self.norm(self.conv1(x)))
        x = F.relu(self.norm(self.conv2(x)))
        x = F.relu(self.norm(self.conv3(x)))
        return x


class UpConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels: number of channels in input (1st) feature map
            out_channels: number of channels in output feature maps
        """
        super(UpConvBlock, self).__init__()
        self.upconv = UpConv2x2(in_channels)
        self.conv1 = conv3x3(in_channels, out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.conv3 = conv3x3(out_channels, out_channels)
        self.norm = nn.BatchNorm2d(out_channels, track_running_stats=False)

    def forward(self, xh, xv):
        """
        Args:
            xh: torch Variable, activations from same resolution feature maps (gray arrow in diagram)
            xv: torch Variable, activations from lower resolution feature maps (green arrow in diagram)
        """
        xv = self.upconv(xv)
        x = concat(xh, xv)
        x = F.relu(self.norm(self.conv1(x)))
        x = F.relu(self.norm(self.conv2(x)))
        x = F.relu(self.norm(self.conv3(x)))
        return x


class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()
        fs = [16, 32, 64, 128, 256]
        self.conv_in = ConvBlock(1, fs[0])
        self.dconv1 = DownConvBlock(fs[0], fs[1])
        self.dconv2 = DownConvBlock(fs[1], fs[2])
        self.dconv3 = DownConvBlock(fs[2], fs[3])
        self.dconv4 = DownConvBlock(fs[3], fs[4])
        self.uconv1 = UpConvBlock(fs[4], fs[3])
        self.uconv2 = UpConvBlock(fs[3], fs[2])
        self.uconv3 = UpConvBlock(fs[2], fs[1])
        self.uconv4 = UpConvBlock(fs[1], fs[0])
        self.conv_out = conv3x3(fs[0], 1)
        self._initialize_weights()

    def forward(self, x):
        x1 = self.conv_in(x)
        x2 = self.dconv1(x1)
        x3 = self.dconv2(x2)
        x4 = self.dconv3(x3)
        x5 = self.dconv4(x4)
        x6 = self.uconv1(x4, x5)
        x7 = self.uconv2(x3, x6)
        x8 = self.uconv3(x2, x7)
        x9 = self.uconv4(x1, x8)
        x10 = self.conv_out(x9)
        return x10

    def _initialize_weights(self):
        conv_modules = [m for m in self.modules() if isinstance(m, nn.Conv2d)]
        for m in conv_modules:
            n = m.weight.shape[1] * m.weight.shape[2] * m.weight.shape[3]
            m.weight.data.normal_(0, np.sqrt(2.0 / n))
            m.bias.data.zero_()


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
