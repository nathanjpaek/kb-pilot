import torch
import torch.nn as nn


class Sine(nn.Module):
    """Sine Activation Function."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(30.0 * x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
