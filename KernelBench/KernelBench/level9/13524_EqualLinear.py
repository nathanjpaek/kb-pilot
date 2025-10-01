import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    rest_dim = [1] * (input.ndim - bias.ndim - 1)
    return F.leaky_relu(input + bias.view(1, bias.shape[0], *rest_dim),
        negative_slope=negative_slope) * scale


class EqualLinear(nn.Module):

    def __init__(self, in_dim, out_dim, bias_init=0, lr_mul=1, activation=None
        ):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.activation = activation
        self.scale = 1 / math.sqrt(in_dim) * lr_mul
        self.lr_mul = lr_mul
        self.bias_init = bias_init

    def forward(self, input):
        bias = self.bias * self.lr_mul + self.bias_init
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, bias)
        else:
            out = F.linear(input, self.weight * self.scale, bias=bias)
        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
            )


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4}]
