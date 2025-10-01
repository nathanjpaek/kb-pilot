import torch
from torch import nn


class exponential(nn.Module):

    def __init__(self):
        super(exponential, self).__init__()

    def forward(self, x):
        return torch.exp(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
