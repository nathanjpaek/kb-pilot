import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel, stride, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel,
            stride=stride, padding=padding)
        self.norm = nn.GroupNorm(1, out_channels)

    def forward(self, x):
        return F.elu(self.norm(self.conv(x)))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel': 4, 'stride': 1}
        ]
