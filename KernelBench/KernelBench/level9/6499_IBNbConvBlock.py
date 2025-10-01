import torch
import torch.nn as nn
import torch.utils.data


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


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4,
        'stride': 1, 'padding': 4}]
