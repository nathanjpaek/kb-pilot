import torch
import torch.utils.data
import torch.nn as nn


def wrn_conv1x1(in_channels, out_channels, stride, activate):
    """
    1x1 version of the WRN specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    activate : bool
        Whether activate the convolution block.
    """
    return WRNConv(in_channels=in_channels, out_channels=out_channels,
        kernel_size=1, stride=stride, padding=0, activate=activate)


def wrn_conv3x3(in_channels, out_channels, stride, activate):
    """
    3x3 version of the WRN specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    activate : bool
        Whether activate the convolution block.
    """
    return WRNConv(in_channels=in_channels, out_channels=out_channels,
        kernel_size=3, stride=stride, padding=1, activate=activate)


class WRNConv(nn.Module):
    """
    WRN specific convolution block.

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
    activate : bool
        Whether activate the convolution block.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        padding, activate):
        super(WRNConv, self).__init__()
        self.activate = activate
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=
            out_channels, kernel_size=kernel_size, stride=stride, padding=
            padding, bias=True)
        if self.activate:
            self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.activate:
            x = self.activ(x)
        return x


class WRNBottleneck(nn.Module):
    """
    WRN bottleneck block for residual path in WRN unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    width_factor : float
        Wide scale factor for width of layers.
    """

    def __init__(self, in_channels, out_channels, stride, width_factor):
        super(WRNBottleneck, self).__init__()
        mid_channels = int(round(out_channels // 4 * width_factor))
        self.conv1 = wrn_conv1x1(in_channels=in_channels, out_channels=
            mid_channels, stride=1, activate=True)
        self.conv2 = wrn_conv3x3(in_channels=mid_channels, out_channels=
            mid_channels, stride=stride, activate=True)
        self.conv3 = wrn_conv1x1(in_channels=mid_channels, out_channels=
            out_channels, stride=1, activate=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class WRNUnit(nn.Module):
    """
    WRN unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    width_factor : float
        Wide scale factor for width of layers.
    """

    def __init__(self, in_channels, out_channels, stride, width_factor):
        super(WRNUnit, self).__init__()
        self.resize_identity = in_channels != out_channels or stride != 1
        self.body = WRNBottleneck(in_channels=in_channels, out_channels=
            out_channels, stride=stride, width_factor=width_factor)
        if self.resize_identity:
            self.identity_conv = wrn_conv1x1(in_channels=in_channels,
                out_channels=out_channels, stride=stride, activate=False)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = x + identity
        x = self.activ(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'stride': 1,
        'width_factor': 4}]
