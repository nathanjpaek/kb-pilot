import torch
from torch import nn as nn


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


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
