import torch
from torch import nn


class HardMish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / 2 * torch.clamp(x + 2, min=0, max=2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
