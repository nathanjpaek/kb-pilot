import torch
from torch import nn


class Upsample4x(nn.Module):

    def __init__(self, n_channels):
        super(Upsample4x, self).__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, 3, 1, 1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=4, mode=
            'bilinear', align_corners=False)
        x = torch.relu(self.conv(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_channels': 4}]
