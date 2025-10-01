import torch
import torch.nn as nn


class UpSample(nn.Module):

    def __init__(self, n_channels, factor=2):
        super(UpSample, self).__init__()
        out_channels = n_channels * factor * factor
        self.proj = nn.Conv2d(n_channels, out_channels, 1, 1, 0)
        self.up = nn.PixelShuffle(factor)
        self.init_weight()

    def forward(self, x):
        x = self.proj(x)
        x = self.up(x)
        return x

    def init_weight(self):
        nn.init.xavier_normal_(self.proj.weight, gain=1.0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_channels': 4}]
