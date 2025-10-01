import torch
from torch import nn
import torch.utils.data


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


class CausalConv2d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride=1,
        padding='downright', activation=None):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 2
        self.kernel_size = kernel_size
        if padding == 'downright':
            pad = [kernel_size[1] - 1, 0, kernel_size[0] - 1, 0]
        elif padding == 'down' or padding == 'causal':
            pad = kernel_size[1] // 2
            pad = [pad, pad, kernel_size[0] - 1, 0]
        self.causal = 0
        if padding == 'causal':
            self.causal = kernel_size[1] // 2
        self.pad = nn.ZeroPad2d(pad)
        self.conv = WNConv2d(in_channel, out_channel, kernel_size, stride=
            stride, padding=0, activation=activation)

    def forward(self, input):
        out = self.pad(input)
        if self.causal > 0:
            self.conv.conv.weight_v.data[:, :, -1, self.causal:].zero_()
        out = self.conv(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channel': 4, 'out_channel': 4, 'kernel_size': 4}]
