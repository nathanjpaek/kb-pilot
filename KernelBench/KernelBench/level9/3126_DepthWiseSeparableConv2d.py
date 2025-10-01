import torch
import torch.nn as nn
import torch.jit
import torch.nn


class DepthWiseSeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=1, dilation=1, bias=True):
        """Depthwise separable 2D convolution.

    Args:
      in_channels (int): number of input channels.
      out_channels (int): number of output channels.
      kernel_size (int or (int, int)): kernel size.
      kwargs: additional keyword arguments. See `Conv2d` for details. 
    """
        super(DepthWiseSeparableConv2d, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.point_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, input):
        return self.point_conv(self.depth_conv(input))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
