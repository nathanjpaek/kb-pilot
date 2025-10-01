import torch
from torch import nn
import torch.nn.functional as F


class ResNetBottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, bottleneck_channels,
        stride, downsample=None):
        super(ResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels,
            kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels,
            kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels,
            kernel_size=1, bias=False)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x), inplace=True)
        out = F.relu(self.conv2(out), inplace=True)
        out = self.conv3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = F.relu(out, inplace=True)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'bottleneck_channels':
        4, 'stride': 1}]
