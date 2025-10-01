import torch
import torch.nn as nn


class DepthWiseSeparableConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, bias=True, padding_mode='zeros',
        inner_kernel_size=1, inner_stride=1, inner_padding=0):
        """Depthwise separable 2D Convolution.

        :param in_channels: Input channels.
        :type in_channels: int
        :param out_channels: Output channels.
        :type out_channels: int
        :param kernel_size: Kernel shape/size.
        :type kernel_size: int|tuple|list
        :param stride: Stride.
        :type stride: int|tuple|list
        :param padding: Padding.
        :type padding: int|tuple|list
        :param dilation: Dilation.
        :type dilation: int
        :param bias: Bias.
        :type bias: bool
        :param padding_mode: Padding mode.
        :type padding_mode: str
        :param inner_kernel_size: Kernel shape/size of the second convolution.
        :type inner_kernel_size: int|tuple|list
        :param inner_stride: Inner stride.
        :type inner_stride: int|tuple|list
        :param inner_padding: Inner padding.
        :type inner_padding: int|tuple|list
        """
        super(DepthWiseSeparableConvBlock, self).__init__()
        self.depth_wise_conv: 'nn.Module' = nn.Conv2d(in_channels=
            in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=
            in_channels if out_channels >= in_channels else out_channels,
            bias=bias, padding_mode=padding_mode)
        self.non_linearity: 'nn.Module' = nn.LeakyReLU()
        self.point_wise: 'nn.Module' = nn.Conv2d(in_channels=out_channels,
            out_channels=out_channels, kernel_size=inner_kernel_size,
            stride=inner_stride, padding=inner_padding, dilation=1, groups=
            1, bias=bias, padding_mode=padding_mode)
        if inner_kernel_size != 1:
            None
            None
            raise ValueError
        self.layers = nn.Sequential(self.depth_wise_conv, self.
            non_linearity, self.point_wise)

    def forward(self, x):
        """Forward pass of the module.

        :param x: Input tensor.
        :type x: torch.Tensor
        :return: Output tensor.
        :rtype: torch.Tensor
        """
        return self.layers(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
