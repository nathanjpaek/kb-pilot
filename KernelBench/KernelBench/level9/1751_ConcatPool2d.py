import torch
import torch.nn as nn


class ConcatPool2d(nn.Module):
    """Layer that concats `AvgPool2d` and `MaxPool2d`"""

    def __init__(self, ks, stride=None, padding=0):
        super().__init__()
        self.ap = nn.AvgPool2d(ks, stride, padding)
        self.mp = nn.MaxPool2d(ks, stride, padding)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'ks': 4}]
