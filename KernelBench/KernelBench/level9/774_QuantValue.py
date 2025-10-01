import torch
import torch.utils.data
import torch
from torch import nn
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed


class QuantValue_F(torch.autograd.Function):
    """
    res = clamp(round(input/pow(2,-m)) * pow(2, -m), -pow(2, N-1), pow(2, N-1) - 1)
    """

    @staticmethod
    def forward(ctx, inputs, N, m):
        Q = pow(2, N - 1) - 1
        delt = pow(2, -m)
        M = (inputs / delt).round().clamp(-Q - 1, Q)
        return delt * M

    @staticmethod
    def backward(ctx, g):
        return g, None, None


class QuantValue(nn.Module):
    """
    Quantization
    """

    def __init__(self, N, m):
        super(QuantValue, self).__init__()
        self.N = N
        self.m = m
        self.quant = QuantValue_F.apply

    def forward(self, x):
        return self.quant(x, self.N, self.m)

    def extra_repr(self):
        s = 'N = %d, m = %d' % (self.N, self.m)
        return s


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'N': 4, 'm': 4}]
