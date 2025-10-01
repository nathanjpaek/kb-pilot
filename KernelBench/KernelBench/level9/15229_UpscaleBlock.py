import torch
import torch.nn as nn


class UpscaleBlock(nn.Module):
    """ Upscaling Block using Pixel Shuffle to increase image dimensions. Used in Generator Network"""
    """
    Pixel shuffle layer
    (Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional
    Neural Network, CVPR17)
    However, while this approach helps, it is still easy for deconvolution to fall into creating artifacts.
    https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels=None, kernel_size=3,
        stride=1, upscale_factor=2):
        super(UpscaleBlock, self).__init__()
        if out_channels:
            out_channels = out_channels
        else:
            out_channels = in_channels * upscale_factor ** 2
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=
            out_channels, kernel_size=kernel_size, stride=stride, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=upscale_factor)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4}]
