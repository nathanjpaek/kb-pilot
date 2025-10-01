import torch
import torch.nn as nn
from torch.nn import functional as F


class SelfGate(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x = self.proj(x)
        x, gate = x.chunk(2, dim=-1)
        x = x * F.mish(gate)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_in': 4, 'dim_out': 4}]
