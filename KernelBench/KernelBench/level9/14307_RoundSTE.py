import torch
from torch import nn


class RoundSTE(nn.Module):

    def __init__(self):
        """
        This module perform element-wise rounding with straight through estimator (STE).
        """
        super(RoundSTE, self).__init__()

    def forward(self, x):
        """
        The forward function of the rounding module

        :param x: Input tensor to be rounded
        :return: A rounded tensor
        """
        x_error = torch.round(x) - x
        return x + x_error.detach()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
