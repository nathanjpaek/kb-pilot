import torch
import torch.nn as nn


class Upsample(nn.Module):

    def __init__(self, n_channels, with_conv=True):
        super(Upsample, self).__init__()
        self.with_conv = with_conv
        self.n_channels = n_channels
        self.conv = nn.Conv2d(self.n_channels, self.n_channels, 3, stride=1,
            padding=1)

    def forward(self, x):
        up = nn.Upsample(scale_factor=2, mode='nearest')
        B, C, H, W = x.shape
        assert C == self.n_channels
        x = up(x)
        assert x.shape == (B, C, 2 * H, 2 * W)
        if self.with_conv:
            x = self.conv(x)
            assert x.shape == (B, C, 2 * H, 2 * W)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_channels': 4}]
