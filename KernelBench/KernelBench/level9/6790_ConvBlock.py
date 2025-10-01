import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.cuda import *


def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
        padding=1, bias=True)


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


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
