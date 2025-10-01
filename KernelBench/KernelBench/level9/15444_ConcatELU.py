import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatELU(nn.Module):
    """Activation function that applies ELU in both direction (inverted and plain).

    Allows non-linearity while providing strong gradients for any input (important for final convolution)
    """

    def forward(self, x):
        return torch.cat([F.elu(x), F.elu(-x)], dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
