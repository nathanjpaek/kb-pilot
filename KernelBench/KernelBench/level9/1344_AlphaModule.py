import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.optim


class AlphaModule(nn.Module):

    def __init__(self, shape):
        super(AlphaModule, self).__init__()
        if not isinstance(shape, tuple):
            shape = shape,
        self.alpha = Parameter(torch.rand(tuple([1] + list(shape))) * 0.1,
            requires_grad=True)

    def forward(self, x):
        return x * self.alpha

    def parameters(self, recurse: 'bool'=True):
        yield self.alpha


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'shape': 4}]
