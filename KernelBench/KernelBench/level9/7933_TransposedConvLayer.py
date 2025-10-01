import torch
from torch.optim import *
import torch.nn as nn


class TransposedConvLayer(nn.Module):
    """
    Transposed convolutional layer to increase spatial resolution (x2) in a decoder.
    Default: bias, ReLU, no downsampling, no batch norm.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
        activation='relu', norm=None):
        super(TransposedConvLayer, self).__init__()
        bias = False if norm == 'BN' else True
        self.transposed_conv2d = nn.ConvTranspose2d(in_channels,
            out_channels, kernel_size, stride=2, padding=padding,
            output_padding=1, bias=bias)
        if activation is not None:
            self.activation = getattr(torch, activation)
        else:
            self.activation = None
        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels,
                track_running_stats=True)

    def forward(self, x):
        out = self.transposed_conv2d(x)
        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)
        if self.activation is not None:
            out = self.activation(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
