import torch
from torch import nn
import torch.utils.data


class Conv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, stride,
        padding, dilation=1, activation=None, bias=True):
        super(Conv2d, self).__init__()
        self.activation = activation
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size,
            stride, padding, dilation, bias=bias)

    def forward(self, x):
        h = self.conv(x)
        if self.activation is None:
            out = h
        else:
            out = self.activation(h)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_channels': 4, 'output_channels': 4, 'kernel_size': 
        4, 'stride': 1, 'padding': 4}]
