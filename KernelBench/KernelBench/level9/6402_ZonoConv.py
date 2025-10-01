import torch
from typing import Tuple
from typing import Union
import torch.utils.data


class ZonoConv(torch.nn.Module):
    """
    Wrapper around pytorch's convolutional layer.
    We only add the bias to the zeroth element of the zonotope
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size:
        'Union[int, Tuple[int, int]]', stride:
        'Union[int, Tuple[int, int]]', dilation:
        'Union[int, Tuple[int, int]]'=1, padding:
        'Union[int, Tuple[int, int]]'=0):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=
            out_channels, kernel_size=kernel_size, stride=stride, dilation=
            dilation, padding=padding, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(out_channels))

    def __call__(self, x: "'torch.Tensor'") ->'torch.Tensor':
        return self.forward(x)

    def forward(self, x: "'torch.Tensor'") ->'torch.Tensor':
        """
        Forward pass through the convolutional layer

        :param x: input zonotope to the convolutional layer.
        :return x: zonotope after being pushed through the convolutional layer.
        """
        x = self.conv(x)
        x = self.zonotope_add(x)
        return x

    def zonotope_add(self, x: "'torch.Tensor'") ->'torch.Tensor':
        """
        Modification required compared to the normal torch conv layers.
        The bias is added only to the central zonotope term and not the error terms.

        :param x: zonotope input to have the bias added.
        :return: zonotope with the bias added to the central (first) term.
        """
        bias = torch.unsqueeze(self.bias, dim=-1)
        bias = torch.unsqueeze(bias, dim=-1)
        x[0] = x[0] + bias
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4,
        'stride': 1}]
