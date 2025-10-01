import torch
import torch.nn as nn
import torch.nn.functional as F


class UpsampleConvLayer(nn.Module):
    """
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReplicationPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4,
        'stride': 1}]
