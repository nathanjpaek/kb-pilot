import torch
import torch.nn as nn


class Conv1d_mp(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size:
        'int', stride: 'int'=1, padding: 'int'=1):
        super(Conv1d_mp, self).__init__()
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._ops = nn.Conv1d(in_channels, out_channels, kernel_size,
            stride, padding)
        self._activation = nn.ReLU()
        self._mp = nn.MaxPool1d(2, 2)

    def forward(self, x):
        return self._mp(self._activation(self._ops(x)))


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
