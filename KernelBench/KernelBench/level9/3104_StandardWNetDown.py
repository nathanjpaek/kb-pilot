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


class StandardWNetDown(nn.Module):

    def __init__(self, in_channels, out_channels, position, activation=nn.
        ReLU()):
        """
    Default down convolution block for the WNet.
    Args:
      in_channels (int): number of input channels.
      out_channels (int): number of output channels.
      position (int): position of the block within the WNet.
    """
        super(StandardWNetDown, self).__init__()
        self.activation = activation
        if position == 0:
            self.block_0 = nn.Conv2d(in_channels, out_channels, 3)
            self.block_1 = nn.Conv2d(in_channels, out_channels, 3)
        else:
            self.block_0 = DepthWiseSeparableConv2d(in_channels,
                out_channels, 3)
            self.block_1 = DepthWiseSeparableConv2d(out_channels,
                out_channels, 3)

    def forward(self, input):
        return self.activation(self.block_1(self.activation(self.block_0(
            input))))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'position': 4}]
