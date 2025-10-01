import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import leaky_relu


def fused_leaky_relu(input_, bias, negative_slope=0.2, scale=2 ** 0.5):
    return scale * leaky_relu(input_ + bias[:input_.shape[1]],
        negative_slope, inplace=True)


class EqualLinear(nn.Module):

    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1.0,
        activation=None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None
        self.activation = activation
        self.scale = 1 / math.sqrt(in_dim) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, x):
        if self.activation:
            out = F.linear(x, self.weight * self.scale)
            if self.activation == 'lrelu':
                out = fused_leaky_relu(out, self.bias * self.lr_mul)
            else:
                raise NotImplementedError
        else:
            out = F.linear(x, self.weight * self.scale, bias=self.bias *
                self.lr_mul)
        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
            )


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4}]
