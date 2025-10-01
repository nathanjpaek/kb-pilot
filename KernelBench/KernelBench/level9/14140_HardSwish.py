import torch
import torch.nn as nn


class HardSwish(nn.Module):

    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()
        self.act = nn.ReLU6(inplace)
    """forward"""

    def forward(self, x):
        return x * self.act(x + 3) / 6


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
