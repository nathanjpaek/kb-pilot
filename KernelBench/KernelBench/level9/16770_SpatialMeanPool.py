import torch
import torch.nn as nn


class SpatialMeanPool(nn.Module):
    """
    Performs mean pooling over spatial dimensions; keeps only the first `ndim`
    dimensions of the input.
    """

    def __init__(self, ndim=2):
        super(SpatialMeanPool, self).__init__()
        self.ndim = ndim

    def forward(self, x):
        return x.mean(tuple(range(self.ndim, x.ndim)))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
