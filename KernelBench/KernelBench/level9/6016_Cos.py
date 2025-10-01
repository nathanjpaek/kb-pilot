import torch
import torch.nn as nn


class Cos(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, X: 'torch.Tensor'):
        return torch.cos(X)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
