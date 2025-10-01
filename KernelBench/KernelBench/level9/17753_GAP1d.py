import torch
from torch import nn
import torch.nn.functional


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class GAP1d(nn.Module):
    """Global Adaptive Pooling + Flatten
    """

    def __init__(self, output_size=1):
        super(GAP1d, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(output_size)
        self.flatten = Flatten()

    def forward(self, x):
        return self.flatten(self.gap(x))


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
