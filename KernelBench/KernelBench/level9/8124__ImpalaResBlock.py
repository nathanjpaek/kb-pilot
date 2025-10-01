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


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_channels': 4}]
