import torch
from torch import nn


class WNConv2d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride=1,
        padding=0, bias=True, activation=None):
        super().__init__()
        self.conv = nn.utils.weight_norm(nn.Conv2d(in_channel, out_channel,
            kernel_size, stride=stride, padding=padding, bias=bias))
        self.out_channel = out_channel
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        self.kernel_size = kernel_size
        self.activation = activation

    def forward(self, input):
        out = self.conv(input)
        if self.activation is not None:
            out = self.activation(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channel': 4, 'out_channel': 4, 'kernel_size': 4}]
