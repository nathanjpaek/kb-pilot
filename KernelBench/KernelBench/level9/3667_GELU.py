import torch
import numpy as np
import torch.nn as nn


class GELU(nn.Module):
    """Gaussian Error Linear Unit.

    Dan Hendrycks∗, Kevin Gimpel

    GAUSSIAN ERROR LINEAR UNITS (GELUS), 2016

    Args:
        x: float Tensor to perform activation.
    Returns:
        `x` with the GELU activation applied.
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 *
            torch.pow(x, 3))))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
