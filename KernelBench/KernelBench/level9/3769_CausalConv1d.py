import torch
from torch import nn as nn


class CausalConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1,
        **kwargs):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
            padding=self.pad, **kwargs)

    def forward(self, x: 'torch.Tensor'):
        x = self.conv(x)
        x = x[:, :, :-self.conv.padding[0]].contiguous()
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
