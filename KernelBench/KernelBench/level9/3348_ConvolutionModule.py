import torch
from torch import Tensor
from torch import nn


class Swish(torch.nn.Module):
    """Construct an Swish object."""

    def forward(self, x: 'Tensor') ->Tensor:
        """Return Swich activation function."""
        return x * torch.sigmoid(x)


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model.
    Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/conformer/convolution.py

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.
        bias (bool): Whether to use bias in conv layers (default=True).

    """

    def __init__(self, channels: 'int', kernel_size: 'int', bias: 'bool'=True
        ) ->None:
        """Construct an ConvolutionModule object."""
        super(ConvolutionModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0
        self.pointwise_conv1 = nn.Conv1d(channels, 2 * channels,
            kernel_size=1, stride=1, padding=0, bias=bias)
        self.depthwise_conv = nn.Conv1d(channels, channels, kernel_size,
            stride=1, padding=(kernel_size - 1) // 2, groups=channels, bias
            =bias)
        self.norm = nn.LayerNorm(channels)
        self.pointwise_conv2 = nn.Conv1d(channels, channels, kernel_size=1,
            stride=1, padding=0, bias=bias)
        self.activation = Swish()

    def forward(self, x: 'Tensor') ->Tensor:
        """Compute convolution module.

        Args:
            x: Input tensor (#time, batch, channels).

        Returns:
            Tensor: Output tensor (#time, batch, channels).

        """
        x = x.permute(1, 2, 0)
        x = self.pointwise_conv1(x)
        x = nn.functional.glu(x, dim=1)
        x = self.depthwise_conv(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = x.permute(0, 2, 1)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        return x.permute(2, 0, 1)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'channels': 4, 'kernel_size': 1}]
