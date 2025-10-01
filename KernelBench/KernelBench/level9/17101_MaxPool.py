import torch
import torch.nn as nn


class MaxPool(nn.Module):

    def __init__(self, kernel_size, stride):
        super(MaxPool, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        x = self.pool(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'kernel_size': 4, 'stride': 1}]
