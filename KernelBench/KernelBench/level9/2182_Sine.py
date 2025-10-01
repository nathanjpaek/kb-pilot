import torch
import torch.nn as nn


class Sine(nn.Module):
    """
    A wrapper for PyTorch sine function.
    """

    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    @staticmethod
    def forward(x):
        return torch.sin(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
