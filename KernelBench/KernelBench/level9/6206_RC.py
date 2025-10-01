import torch
import torch.nn as nn
import torch.nn.functional as F


class RC(nn.Module):
    """
    A wrapper class for ReflectionPad2d, Conv2d and an optional relu
    """

    def __init__(self, in_dim, out_dim, kernel_size=3, padding=1,
        activation_function=True):
        super().__init__()
        self.pad = nn.ReflectionPad2d((padding, padding, padding, padding))
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size)
        self.activation_function = activation_function

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        return F.relu(x) if self.activation_function else x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4}]
