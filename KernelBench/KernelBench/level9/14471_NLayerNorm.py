import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter


class NLayerNorm(nn.Module):

    def __init__(self, n_features: 'int', d: 'int') ->None:
        super().__init__()
        self.weight = Parameter(torch.ones(n_features, d))
        self.bias = Parameter(torch.zeros(n_features, d))

    def forward(self, x: 'Tensor') ->Tensor:
        assert x.ndim == 3
        x = (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True)
        x = self.weight * x + self.bias
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'n_features': 4, 'd': 4}]
