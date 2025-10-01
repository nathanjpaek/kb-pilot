import torch
from torch import nn


class MockModule(nn.Module):

    def __init__(self, k: 'int'):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(k, dtype=torch.float))

    def forward(self, x, y=None):
        assert len(x.shape) == 2
        out = x + self.p
        if y is not None:
            out = out + y
        return out


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'k': 4}]
