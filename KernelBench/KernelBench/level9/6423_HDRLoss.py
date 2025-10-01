import torch
from torch import nn
from numpy import *
from math import sqrt as sqrt
from itertools import product as product


class HDRLoss(nn.Module):
    """High dynamic range loss."""

    def __init__(self, eps=0.01):
        """Initializes loss with numerical stability epsilon."""
        super(HDRLoss, self).__init__()
        self._eps = eps

    def forward(self, denoised, target):
        """Computes loss by unpacking render buffer."""
        loss = (denoised - target) ** 2 / (denoised + self._eps) ** 2
        return torch.mean(loss.view(-1))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
