import torch
import torch.nn
import torch.nn as nn
import torch.nn.parallel


class CMul(nn.Module):
    """
    nn.CMul in Torch7.
    """

    def __init__(self):
        super(CMul, self).__init__()

    def forward(self, x):
        return x[0] * x[1]

    def __repr__(self):
        return self.__class__.__name__


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
