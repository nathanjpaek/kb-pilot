import torch
from torch import nn


class _ImpalaResBlock(nn.Module):

    def __init__(self, n_channels: 'int'):
        super().__init__()
        self.n_channels = n_channels
        kernel_size = 3
        padding = 1
        self.relu = nn.ReLU()
        self.relu_inplace = nn.ReLU()
        self.conv1 = nn.Conv2d(n_channels, n_channels, kernel_size, padding
            =padding)
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size, padding
            =padding)

    def forward(self, inputs):
        x = self.relu(inputs)
        x = self.conv1(x)
        x = self.relu_inplace(x)
        x = self.conv2(x)
        x += inputs
        return x


class _ImpalaBlock(nn.Module):

    def __init__(self, n_channels_in: 'int', n_channels_out: 'int'):
        super().__init__()
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        kernel_size = 3
        padding = 1
        self.conv1 = nn.Conv2d(n_channels_in, n_channels_out, kernel_size,
            padding=padding)
        self.pool = nn.MaxPool2d(kernel_size, stride=2, padding=padding)
        self.res1 = _ImpalaResBlock(n_channels_out)
        self.res2 = _ImpalaResBlock(n_channels_out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_channels_in': 4, 'n_channels_out': 4}]
