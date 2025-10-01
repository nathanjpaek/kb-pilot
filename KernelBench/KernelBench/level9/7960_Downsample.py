import torch
import torch.nn as nn


class Downsample(nn.Module):

    def __init__(self, n_channels, with_conv=True):
        super(Downsample, self).__init__()
        self.with_conv = with_conv
        self.n_channels = n_channels
        self.conv = nn.Conv2d(self.n_channels, self.n_channels, 3, stride=2,
            padding=1)

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.n_channels
        if self.with_conv:
            x = self.conv(x)
        else:
            down = nn.AvgPool2d(2)
            x = down(x)
        assert x.shape == (B, C, H // 2, W // 2)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_channels': 4}]
