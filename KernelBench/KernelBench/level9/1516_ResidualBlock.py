import torch
from torch import nn


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()
        self.should_apply_shortcut = self.in_channels != self.out_channels

    def forward(self, x):
        if self.should_apply_shortcut:
            residual = self.shortcut(x)
        else:
            residual = x
        x = self.blocks(x)
        x += residual
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
