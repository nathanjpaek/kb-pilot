import torch
import torch.nn as nn


class PixBlock(nn.Module):

    def __init__(self, in_size, out_size=3, scale=2, norm=None):
        super(PixBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size * 2 ** scale, 1, 1)
        self.up = nn.PixelShuffle(scale)

    def forward(self, x):
        x = self.conv1(x)
        x = self.up(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_size': 4}]
