import torch
import torch.nn as nn


class SimpleConv(nn.Module):

    def __init__(self, in_size):
        super(SimpleConv, self).__init__()
        self.conv = nn.Conv2d(in_size, 6, 3, padding='same')
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_size': 4}]
