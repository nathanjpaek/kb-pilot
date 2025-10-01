import torch
import torch.nn.functional as F
import torch.nn as nn


class ConvNorm(nn.Module):
    """ 1D Convolution """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
        padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            bias=bias)

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class Conv1DBlock(nn.Module):
    """ 1D Convolutional Block """

    def __init__(self, in_channels, out_channels, kernel_size, activation=
        None, dropout=None):
        super(Conv1DBlock, self).__init__()
        self.conv_layer = nn.Sequential()
        self.conv_layer.add_module('conv_layer', ConvNorm(in_channels,
            out_channels, kernel_size=kernel_size, stride=1, padding=int((
            kernel_size - 1) / 2), dilation=1, w_init_gain='tanh'))
        if activation is not None:
            self.conv_layer.add_module('activ', activation)
        self.dropout = dropout

    def forward(self, x, mask=None):
        x = x.contiguous().transpose(1, 2)
        x = self.conv_layer(x)
        if self.dropout is not None:
            x = F.dropout(x, self.dropout, self.training)
        x = x.contiguous().transpose(1, 2)
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
