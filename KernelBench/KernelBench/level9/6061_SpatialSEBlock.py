import torch
from torch import nn


class SpatialSEBlock(nn.Module):

    def __init__(self, channel):
        super(SpatialSEBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=channel, out_channels=1,
            kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.sigmoid(self.conv(x))
        return x * y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channel': 4}]
