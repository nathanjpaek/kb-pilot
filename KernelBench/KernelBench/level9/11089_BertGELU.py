import math
import torch
from torch import nn


class BertGELU(nn.Module):
    """Bert uses GELU as the activation function for the position-wise network.
    """

    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
