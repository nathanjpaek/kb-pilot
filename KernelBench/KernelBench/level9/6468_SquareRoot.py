import torch
import torch.nn.functional
from torch import nn


class SquareRoot(nn.Module):

    def forward(self, x):
        return x.sqrt()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
