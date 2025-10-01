import torch
import torch.utils.data
import torch.nn as nn


class Down2d(nn.Module):
    """docstring for Down2d."""

    def __init__(self, in_channel, out_channel, kernel, stride, padding):
        super(Down2d, self).__init__()
        self.c1 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel,
            stride=stride, padding=padding)
        self.n1 = nn.InstanceNorm2d(out_channel)
        self.c2 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel,
            stride=stride, padding=padding)
        self.n2 = nn.InstanceNorm2d(out_channel)

    def forward(self, x):
        x1 = self.c1(x)
        x1 = self.n1(x1)
        x2 = self.c2(x)
        x2 = self.n2(x2)
        x3 = x1 * torch.sigmoid(x2)
        return x3


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channel': 4, 'out_channel': 4, 'kernel': 4, 'stride': 
        1, 'padding': 4}]
