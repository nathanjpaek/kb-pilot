import torch
import torch.nn as nn


class UpConv2D(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, kernel_size=5, ratio=2):
        super(UpConv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * ratio ** 2,
            kernel_size, padding=kernel_size // 2)
        self.upscale = nn.PixelShuffle(ratio)

    def forward(self, input_):
        x = self.conv(input_)
        output = self.upscale(x)
        return output


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
