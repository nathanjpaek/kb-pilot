import torch
from torch import nn


class ConvShuffle(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding=
        'same', upscale_factor=2, padding_mode='zeros'):
        super(ConvShuffle, self).__init__()
        self.upscale_factor = upscale_factor
        self.conv = nn.Conv2d(in_channels, out_channels * upscale_factor **
            2, kernel_size, padding=padding, padding_mode=padding_mode)

    def forward(self, X):
        X = self.conv(X)
        return nn.functional.pixel_shuffle(X, self.upscale_factor)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
