import torch
import torch.nn as nn


class CausalConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
            stride=stride, padding=kernel_size)

    def forward(self, x):
        x = self.conv(x)
        return x[:, :, :-self.kernel_size]


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
