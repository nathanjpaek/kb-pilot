import torch
from torch import nn


class Maxout(nn.Module):

    def __init__(self, pool_size):
        super().__init__()
        self._pool_size = pool_size

    def forward(self, x):
        assert x.shape[-1
            ] % self._pool_size == 0, 'Wrong input last dim size ({}) for Maxout({})'.format(
            x.shape[-1], self._pool_size)
        m, _i = x.view(*x.shape[:-1], x.shape[-1] // self._pool_size, self.
            _pool_size).max(-1)
        return m


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'pool_size': 4}]
