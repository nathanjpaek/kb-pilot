import torch
from torch import nn as nn
from torch.nn import functional as F


class CausalConv1d(nn.Module):
    """A 1D causal convolution layer.

    Input: (B, D_in, T), where B is the minibatch size, D_in is the number of
        dimensions per step, and T is the number of steps.
    Output: (B, D_out, T), where B is the minibatch size, D_out is the number
        of dimensions in the output, and T is the number of steps.

    Arguments:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
    """

    def __init__(self, in_channels, out_channels, dilation=1):
        super(CausalConv1d, self).__init__()
        self.padding = dilation
        self.causal_conv = nn.Conv1d(in_channels, out_channels, 2, padding=
            self.padding, dilation=dilation)

    def forward(self, minibatch):
        return self.causal_conv(minibatch)[:, :, :-self.padding]


class DenseBlock(nn.Module):
    """Two parallel 1D causal convolution layers w/tanh and sigmoid activations

    Input: (B, D_in, T), where B is the minibatch size, D_in is the number of
        dimensions of the input, and T is the number of steps.
    Output: (B, D_in+F, T), where where `B` is the minibatch size, `D_in` is the
        number of dimensions of the input, `F` is the number of filters, and `T`
        is the length of the input sequence.

    Arguments:
        in_channels (int): number of input channels
        filters (int): number of filters per channel
    """

    def __init__(self, in_channels, filters, dilation=1):
        super(DenseBlock, self).__init__()
        self.causal_conv1 = CausalConv1d(in_channels, filters, dilation=
            dilation)
        self.causal_conv2 = CausalConv1d(in_channels, filters, dilation=
            dilation)

    def forward(self, minibatch):
        tanh = F.tanh(self.causal_conv1(minibatch))
        sig = F.sigmoid(self.causal_conv2(minibatch))
        out = torch.cat([minibatch, tanh * sig], dim=1)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'filters': 4}]
