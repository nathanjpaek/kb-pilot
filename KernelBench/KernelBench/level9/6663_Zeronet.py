import torch
import torch.nn as nn


class Zeronet(nn.Module):

    def forward(self, x):
        """
        Return a zero-out copy of x
        :param x: torch.Tensor
        :return: x*0, type torch.Tensor
        """
        return torch.zeros_like(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
