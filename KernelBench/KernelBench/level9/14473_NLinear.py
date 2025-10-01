import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter


class NLinear(nn.Module):

    def __init__(self, n: 'int', d_in: 'int', d_out: 'int', bias: 'bool'=True
        ) ->None:
        super().__init__()
        self.weight = Parameter(Tensor(n, d_in, d_out))
        self.bias = Parameter(Tensor(n, d_out)) if bias else None
        with torch.no_grad():
            for i in range(n):
                layer = nn.Linear(d_in, d_out)
                self.weight[i] = layer.weight.T
                if self.bias is not None:
                    self.bias[i] = layer.bias

    def forward(self, x):
        assert x.ndim == 3
        x = x[..., None] * self.weight[None]
        x = x.sum(-2)
        if self.bias is not None:
            x = x + self.bias[None]
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'n': 4, 'd_in': 4, 'd_out': 4}]
