import torch
import torch.nn.functional as F
import torch.nn as nn


class UpsampleConv2d(nn.Module):
    """
    Avoid checkerboard patterns by upsampling the image and convolving.

    https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        padding, upsample):
        """Set parameters for upsampling."""
        super(UpsampleConv2d, self).__init__()
        self.upsample = upsample
        self.padding = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        """
        Upsample then convolve the image.

        "We’ve had our best results with nearest-neighbor interpolation, and
        had difficulty making bilinear resize work. This may simply mean that,
        for our models, the nearest-neighbor happened to work well with
        hyper-parameters optimized for deconvolution. It might also point at
        trickier issues with naively using bilinear interpolation, where it
        resists high-frequency image features too strongly. We don’t
        necessarily think that either approach is the final solution to
        upsampling, but they do fix the checkerboard artifacts."
        (https://distill.pub/2016/deconv-checkerboard/)
        """
        x = F.interpolate(x, mode='nearest', scale_factor=self.upsample)
        return self.conv(self.padding(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4,
        'stride': 1, 'padding': 4, 'upsample': 4}]
