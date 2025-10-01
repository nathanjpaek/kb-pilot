import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class ln_mod(nn.Module):

    def __init__(self, nx, eps=1e-05):
        super().__init__()
        self.eps = eps
        self.weight = Parameter(torch.Tensor(nx))

    def forward(self, x):
        return x / torch.sqrt(torch.std(x, axis=-1, unbiased=False, keepdim
            =True) ** 2 + self.eps) * self.weight.data[..., :]


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nx': 4}]
