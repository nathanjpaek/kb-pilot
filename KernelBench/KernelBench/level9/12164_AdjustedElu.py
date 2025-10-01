import torch
from torch import nn
from torch.nn import functional as F


class AdjustedElu(nn.Module):
    """
    Elu activation function that's adjusted to:
    1) ensure that all outputs are positive and
    2) f(x) = x for x >= 1
    """

    def forward(self, x):
        return F.elu(x - 1.0) + 1.0


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
