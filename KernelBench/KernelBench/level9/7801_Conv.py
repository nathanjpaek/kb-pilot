import torch
import torch.nn as nn


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module


class Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
        padding=1, transpose=False, use_spectral_norm=False):
        super(Conv, self).__init__()
        self.out_channels = out_channels
        if transpose:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding,
                bias=not use_spectral_norm)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=
                kernel_size, stride=stride, padding=padding, bias=not
                use_spectral_norm)
        if use_spectral_norm:
            self.conv = spectral_norm(self.conv)

    def forward(self, x):
        return self.conv(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
