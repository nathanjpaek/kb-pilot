import torch
import torch.nn as nn


class GatedConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(GatedConv2d, self).__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels, out_channels * 3, kernel_size=
            kernel_size, padding=padding)

    def forward(self, x):
        h = self.conv(x)
        a, b, c = torch.chunk(h, chunks=3, dim=1)
        return a + b * torch.sigmoid(c)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4,
        'padding': 4}]
