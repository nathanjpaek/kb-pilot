import torch
import torch.nn as nn


class SpatialMaxPool(nn.Module):
    """
    Performs max pooling over spatial dimensions; keeps only the first `ndim`
    dimensions of the input.
    """

    def __init__(self, ndim=2):
        super(SpatialMaxPool, self).__init__()
        self.ndim = ndim

    def forward(self, x):
        max, _argmax = x.flatten(self.ndim).max(dim=-1)
        return max


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
