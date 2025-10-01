import torch
from torch import Tensor
from torch import nn


class GELU(nn.Module):
    """Quick GELU"""

    def forward(self, x: 'Tensor') ->Tensor:
        return x * torch.sigmoid(1.702 * x)


class MLP(nn.Module):

    def __init__(self, c1, ch, c2=None):
        super().__init__()
        self.c_fc = nn.Linear(c1, ch)
        self.gelu = GELU()
        self.c_proj = nn.Linear(ch, c2 or c1)

    def forward(self, x: 'Tensor') ->Tensor:
        return self.c_proj(self.gelu(self.c_fc(x)))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'c1': 4, 'ch': 4}]
