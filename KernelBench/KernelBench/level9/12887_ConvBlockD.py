import torch
import torch.nn as nn


class ConvBlockD(nn.Module):

    def __init__(self, in_channels, out_channels, groups=3, ker_size=2):
        super(ConvBlockD, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups

        def wn(x):
            return torch.nn.utils.weight_norm(x)
        self.group_conv = wn(nn.Conv2d(self.in_channels, self.in_channels, 
            1, groups=self.groups))
        self.depth_conv = wn(nn.Conv2d(self.in_channels, self.in_channels, 
            3, padding=ker_size, dilation=ker_size, groups=in_channels))
        self.point_conv = wn(nn.Conv2d(self.in_channels, self.out_channels,
            1, groups=1))

    def forward(self, x):
        x = self.group_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


def get_inputs():
    return [torch.rand([4, 18, 64, 64])]


def get_init_inputs():
    return [[], {'in_channels': 18, 'out_channels': 4}]
