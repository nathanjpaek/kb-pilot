import torch
import torch.nn as nn


def conv1x1(in_channels, out_channels, stride=1, groups=1, bias=False):
    """
    Convolution 1x1 layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    """
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
        kernel_size=1, stride=stride, groups=groups, bias=bias)


def conv3x3(in_channels, out_channels, stride=1, padding=1, dilation=1,
    groups=1, bias=False):
    """
    Convolution 3x3 layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    """
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
        kernel_size=3, stride=stride, padding=padding, dilation=dilation,
        groups=groups, bias=bias)


class Flatten(nn.Module):
    """
    Simple flatten module.
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


class NavigatorBranch(nn.Module):
    """
    Navigator branch block for Navigator unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    """

    def __init__(self, in_channels, out_channels, stride):
        super(NavigatorBranch, self).__init__()
        mid_channels = 128
        self.down_conv = conv3x3(in_channels=in_channels, out_channels=
            mid_channels, stride=stride, bias=True)
        self.activ = nn.ReLU(inplace=False)
        self.tidy_conv = conv1x1(in_channels=mid_channels, out_channels=
            out_channels, bias=True)
        self.flatten = Flatten()

    def forward(self, x):
        y = self.down_conv(x)
        y = self.activ(y)
        z = self.tidy_conv(y)
        z = self.flatten(z)
        return z, y


class NavigatorUnit(nn.Module):
    """
    Navigator init.
    """

    def __init__(self):
        super(NavigatorUnit, self).__init__()
        self.branch1 = NavigatorBranch(in_channels=2048, out_channels=6,
            stride=1)
        self.branch2 = NavigatorBranch(in_channels=128, out_channels=6,
            stride=2)
        self.branch3 = NavigatorBranch(in_channels=128, out_channels=9,
            stride=2)

    def forward(self, x):
        t1, x = self.branch1(x)
        t2, x = self.branch2(x)
        t3, _ = self.branch3(x)
        return torch.cat((t1, t2, t3), dim=1)


def get_inputs():
    return [torch.rand([4, 2048, 64, 64])]


def get_init_inputs():
    return [[], {}]
