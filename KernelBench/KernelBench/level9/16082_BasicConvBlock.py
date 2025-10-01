import torch
from torch import nn
import torch.nn.functional as F


class BasicConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(BasicConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
            padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
            padding=1)
        self.norm = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x = F.leaky_relu(self.norm(self.conv2(self.conv1(x))),
            negative_slope=0.001, inplace=True)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
