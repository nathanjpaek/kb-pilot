import torch
import torch.nn as nn


class ConvUnit(nn.Module):

    def __init__(self, in_channels):
        super(ConvUnit, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=in_channels, out_channels=32,
            kernel_size=9, stride=2, bias=True)

    def forward(self, x):
        return self.conv0(x)


def get_inputs():
    return [torch.rand([4, 4, 64, 64])]


def get_init_inputs():
    return [[], {'in_channels': 4}]
