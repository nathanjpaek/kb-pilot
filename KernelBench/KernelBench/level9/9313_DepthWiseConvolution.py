import torch
from torch import nn as nn


class DepthWiseConvolution(nn.Module):

    def __init__(self, channels, kernelSize, stride, expansionFactor):
        super(DepthWiseConvolution, self).__init__()
        channels = channels * expansionFactor
        self.layer = nn.Conv2d(channels, channels, kernelSize, stride, (
            kernelSize - 1) // 2, groups=channels, bias=True)

    def forward(self, x):
        return self.layer(x)


def get_inputs():
    return [torch.rand([4, 16, 64, 64])]


def get_init_inputs():
    return [[], {'channels': 4, 'kernelSize': 4, 'stride': 1,
        'expansionFactor': 4}]
