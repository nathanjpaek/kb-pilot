import torch
import torch.nn as nn


class SubtractMedian(nn.Module):
    """
    Subtracts the median over the last axis.
    """

    def forward(self, x):
        return x - x.median(-1, keepdim=True).values


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
