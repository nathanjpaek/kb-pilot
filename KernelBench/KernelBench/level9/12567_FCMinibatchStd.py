import math
import torch
from torch import nn
from torch.nn import functional as F


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    rest_dim = [1] * (input.ndim - bias.ndim - 1)
    if input.ndim == 3:
        return F.leaky_relu(input + bias.view(1, *rest_dim, bias.shape[0]),
            negative_slope=negative_slope) * scale
    else:
        return F.leaky_relu(input + bias.view(1, bias.shape[0], *rest_dim),
            negative_slope=negative_slope) * scale


class EqualLinear(nn.Module):

    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1,
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

    def forward(self, input):
        bias = self.bias * self.lr_mul if self.bias is not None else None
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


class FCMinibatchStd(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.fc = EqualLinear(in_channel + 1, out_channel, activation=
            'fused_lrelu')

    def forward(self, out):
        stddev = torch.sqrt(out.var(0, unbiased=False) + 1e-08).mean().view(
            1, 1).repeat(out.size(0), 1)
        out = torch.cat([out, stddev], 1)
        out = self.fc(out)
        return out


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_channel': 4, 'out_channel': 4}]
