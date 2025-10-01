import torch
import torch.nn as nn


class SquareActivation(nn.Module):
    """
    Square activation function, clamps the output between 0 and 20 to avoid overflow
    """

    @staticmethod
    def forward(x):
        return torch.clamp(x ** 2, 0, 20)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
