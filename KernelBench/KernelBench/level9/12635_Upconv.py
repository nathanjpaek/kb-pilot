import math
import torch
import torch.nn.functional as F
from torch.nn import Conv2d
from torch.nn import Upsample


class PadSameConv2d(torch.nn.Module):

    def __init__(self, kernel_size, stride=1):
        """
        Imitates padding_mode="same" from tensorflow.
        :param kernel_size: Kernelsize of the convolution, int or tuple/list
        :param stride: Stride of the convolution, int or tuple/list
        """
        super().__init__()
        if isinstance(kernel_size, (tuple, list)):
            self.kernel_size_y = kernel_size[0]
            self.kernel_size_x = kernel_size[1]
        else:
            self.kernel_size_y = kernel_size
            self.kernel_size_x = kernel_size
        if isinstance(stride, (tuple, list)):
            self.stride_y = stride[0]
            self.stride_x = stride[1]
        else:
            self.stride_y = stride
            self.stride_x = stride

    def forward(self, x: 'torch.Tensor'):
        _, _, height, width = x.shape
        padding_y = (self.stride_y * (math.ceil(height / self.stride_y) - 1
            ) + self.kernel_size_y - height) / 2
        padding_x = (self.stride_x * (math.ceil(width / self.stride_x) - 1) +
            self.kernel_size_x - width) / 2
        padding = [math.floor(padding_x), math.ceil(padding_x), math.floor(
            padding_y), math.ceil(padding_y)]
        return F.pad(input=x, pad=padding)


class Upconv(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        """
        Performs two convolutions and a leaky relu. The first operation only convolves in y direction, the second one
        only in x direction.
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Kernel size for the convolutions, first in y direction, then in x direction
        :param stride: Stride for the convolutions, first in y direction, then in x direction
        """
        super().__init__()
        self.upsample = Upsample(scale_factor=2)
        self.pad = PadSameConv2d(kernel_size=2)
        self.conv = Conv2d(in_channels=in_channels, out_channels=
            out_channels, kernel_size=2, stride=1)

    def forward(self, x: 'torch.Tensor'):
        t = self.upsample(x)
        t = self.pad(t)
        return self.conv(t)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
