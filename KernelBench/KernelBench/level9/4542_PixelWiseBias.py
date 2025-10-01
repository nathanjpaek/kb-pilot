import torch
import torch.nn as nn


class PixelWiseBias(nn.Module):
    """Some Information about PixelWiseBias"""

    def __init__(self, channels):
        super(PixelWiseBias, self).__init__()
        self.channels = channels
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x + self.bias[None, :, None, None]
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channels': 4}]
