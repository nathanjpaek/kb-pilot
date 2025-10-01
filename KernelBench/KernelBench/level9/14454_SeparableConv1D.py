import torch
from torch import nn


class SeparableConv1D(nn.Module):
    """Depthwise separable 1D convolution.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution (default 1).
        dilation (int): Spacing between kernel elements (default 1).
        padding (int): Zero-padding added to both sides of the input.
        padding_mode (str): 'zeros', 'reflect', 'replicate' or 'circular' (default 'zeros').
        bias (bool): If True, adds a learnable bias to the output (default: True).

    """

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size:
        'int', stride: 'int'=1, dilation: 'int'=1, padding: 'int'=0,
        padding_mode: 'str'='zeros', bias: 'bool'=True):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=
            kernel_size, stride=stride, dilation=dilation, padding=padding,
            padding_mode=padding_mode, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor (batch, time, channels).

        Returns:
            torch.Tensor: Output tensor (batch, time, channels).
        """
        x = x.transpose(1, -1)
        x = self.pointwise(self.depthwise(x)).transpose(1, -1)
        return x


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
