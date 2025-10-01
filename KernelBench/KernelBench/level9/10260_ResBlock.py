import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init as init


class conv_relu(nn.Module):
    """docstring for conv_relu"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_relu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)

    def forward(self, x):
        out = F.relu(self.conv(x), inplace=True)
        return out


class ResBlock(nn.Module):
    """docstring for ResBlock"""

    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.res1a = conv_relu(in_channels, 128, kernel_size=1)
        self.res1b = conv_relu(128, 128, kernel_size=3, padding=1)
        self.res1c = conv_relu(128, 256, kernel_size=1)
        self.res2a = conv_relu(in_channels, 256, kernel_size=1)

    def forward(self, x):
        out1 = self.res1a(x)
        out1 = self.res1b(out1)
        out1 = self.res1c(out1)
        out2 = self.res2a(x)
        out = out1 + out2
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4}]
