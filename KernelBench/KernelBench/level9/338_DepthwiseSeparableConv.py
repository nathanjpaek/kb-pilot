import torch
from torch import nn


class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_channels, output_channels, kernel_size, padding=0,
        kernels_per_layer=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels *
            kernels_per_layer, kernel_size=kernel_size, padding=padding,
            groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels * kernels_per_layer,
            output_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'output_channels': 4, 'kernel_size': 4}]
