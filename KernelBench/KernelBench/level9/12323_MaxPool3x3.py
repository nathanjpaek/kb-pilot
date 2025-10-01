import torch
import torch.nn as nn


class MaxPool3x3(nn.Module):
    """3x3 max pool with no subsampling."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
        padding=1):
        super(MaxPool3x3, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size, stride, padding)

    def forward(self, x):
        x = self.maxpool(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
