import torch
from torch import nn
import torch.nn.functional as F


class EqualizedLinear(nn.Module):

    def __init__(self, input_size, output_size, gain=2 ** 0.5, lrmul=0.01):
        super().__init__()
        he_std = gain * input_size ** -0.5
        init_std = 1.0 / lrmul
        self.w_mul = he_std * lrmul
        self.weight = torch.nn.Parameter(torch.randn(output_size,
            input_size) * init_std)
        self.bias = torch.nn.Parameter(torch.zeros(output_size))
        self.b_mul = lrmul

    def forward(self, x):
        bias = self.bias
        bias = bias * self.b_mul
        return F.linear(x, self.weight * self.w_mul, bias)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'output_size': 4}]
