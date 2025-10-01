import torch
from torch import nn


class ShuffleConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding=
        'same', upscale_factor=2, padding_mode='zeros'):
        super(ShuffleConv, self).__init__()
        self.upscale_factor = upscale_factor
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            padding=padding, padding_mode=padding_mode)

    def forward(self, X):
        X = torch.cat([X] * self.upscale_factor ** 2, dim=1)
        X = nn.functional.pixel_shuffle(X, self.upscale_factor)
        return self.conv(X)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
