import torch
import torch.utils.data
import torch.nn as nn


class DiracConv(nn.Module):
    """
    DiracNetV2 specific convolution block with pre-activation.

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
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding
        ):
        super(DiracConv, self).__init__()
        self.activ = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=
            out_channels, kernel_size=kernel_size, stride=stride, padding=
            padding, bias=True)

    def forward(self, x):
        x = self.activ(x)
        x = self.conv(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4,
        'stride': 1, 'padding': 4}]
