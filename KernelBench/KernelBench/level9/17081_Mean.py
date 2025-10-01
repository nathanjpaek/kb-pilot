import torch
from typing import Optional
from torch import nn


class Mean(nn.Module):

    def __init__(self, dim: 'Optional[int]'=None, keepdim: 'bool'=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        return torch.mean(input, dim=self.dim, keepdim=self.keepdim)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
