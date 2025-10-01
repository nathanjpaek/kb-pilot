import torch
from torch import nn as nn


class PointWiseConvolution(nn.Module):

    def __init__(self, inChannels, outChannels, stride, expansionFactor,
        isNormal):
        super(PointWiseConvolution, self).__init__()
        if isNormal:
            self.layer = nn.Conv2d(in_channels=inChannels * expansionFactor,
                out_channels=outChannels, kernel_size=1, stride=stride,
                bias=True)
        else:
            self.layer = nn.Conv2d(in_channels=inChannels, out_channels=
                inChannels * expansionFactor, kernel_size=1, stride=stride,
                bias=True)

    def forward(self, x):
        return self.layer(x)


def get_inputs():
    return [torch.rand([4, 16, 64, 64])]


def get_init_inputs():
    return [[], {'inChannels': 4, 'outChannels': 4, 'stride': 1,
        'expansionFactor': 4, 'isNormal': 4}]
