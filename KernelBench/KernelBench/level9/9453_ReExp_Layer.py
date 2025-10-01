import torch
import torch.nn as nn


class ReExp_Layer(nn.Module):
    """
    Description:
        A modified exponential layer.
        Only the negative part of the exponential retains.
        The positive part is linear: y=x+1.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        l = nn.ELU()
        return torch.add(l(x), 1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
