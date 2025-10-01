import torch
import torch.nn as nn


class PixelwiseNorm(nn.Module):
    """
    layer pixelwise normalization
    """

    def __init__(self, eps=1e-07):
        super(PixelwiseNorm, self).__init__()
        self.eps = eps

    def forward(self, x):
        return x / torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True) + self.eps
            )


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
