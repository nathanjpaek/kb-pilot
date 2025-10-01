import torch
import torch.nn as nn


class Conv2D(nn.Module):
    """
    2D convolution with GroupNorm and ELU

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_size : int
        Kernel size
    stride : int
        Stride
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_base = nn.Conv2d(in_channels, out_channels, kernel_size=
            kernel_size, stride=stride)
        self.pad = nn.ConstantPad2d([kernel_size // 2] * 4, value=0)
        self.normalize = torch.nn.GroupNorm(16, out_channels)
        self.activ = nn.ELU(inplace=True)

    def forward(self, x):
        """Runs the Conv2D layer."""
        x = self.conv_base(self.pad(x))
        return self.activ(self.normalize(x))


class UnpackLayerConv2d(nn.Module):
    """
    Unpacking layer with 2d convolutions. Takes a [B,C,H,W] tensor, convolves it
    to produce [B,(r^2)C,H,W] and then unpacks it to produce [B,C,rH,rW].
    """

    def __init__(self, in_channels, out_channels, kernel_size, r=2):
        """
        Initializes a UnpackLayerConv2d object.

        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        kernel_size : int
            Kernel size
        r : int
            Packing ratio
        """
        super().__init__()
        self.conv = Conv2D(in_channels, out_channels * r ** 2, kernel_size, 1)
        self.unpack = nn.PixelShuffle(r)

    def forward(self, x):
        """Runs the UnpackLayerConv2d layer."""
        x = self.conv(x)
        x = self.unpack(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
