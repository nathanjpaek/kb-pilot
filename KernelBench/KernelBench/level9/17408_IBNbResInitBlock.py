import torch
import torch.utils.data
import torch.nn as nn


def ibnb_conv7x7_block(in_channels, out_channels, stride=1, padding=3, bias
    =False, activate=True):
    """
    7x7 version of the IBN(b)-ResNet specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 3
        Padding value for convolution layer.
    bias : bool, default False
        Whether the layer uses a bias vector.
    activate : bool, default True
        Whether activate the convolution block.
    """
    return IBNbConvBlock(in_channels=in_channels, out_channels=out_channels,
        kernel_size=7, stride=stride, padding=padding, bias=bias, activate=
        activate)


class IBNbConvBlock(nn.Module):
    """
    IBN(b)-ResNet specific convolution block with Instance normalization and ReLU activation.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    activate : bool, default True
        Whether activate the convolution block.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        padding, dilation=1, groups=1, bias=False, activate=True):
        super(IBNbConvBlock, self).__init__()
        self.activate = activate
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=
            out_channels, kernel_size=kernel_size, stride=stride, padding=
            padding, dilation=dilation, groups=groups, bias=bias)
        self.inst_norm = nn.InstanceNorm2d(num_features=out_channels,
            affine=True)
        if self.activate:
            self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.inst_norm(x)
        if self.activate:
            x = self.activ(x)
        return x


class IBNbResInitBlock(nn.Module):
    """
    IBN(b)-ResNet specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        super(IBNbResInitBlock, self).__init__()
        self.conv = ibnb_conv7x7_block(in_channels=in_channels,
            out_channels=out_channels, stride=2)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
