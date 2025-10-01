import math
import torch
import torch as th
import torch.nn as nn


class PELU(nn.Module):

    def __init__(self, a=None, b=None):
        super().__init__()
        default_val = math.sqrt(0.1)
        a = default_val if a is None else a
        b = default_val if b is None else b
        self.a = nn.Parameter(th.tensor(a), requires_grad=True)
        self.b = nn.Parameter(th.tensor(b), requires_grad=True)

    def forward(self, inputs):
        a = th.abs(self.a)
        b = th.abs(self.b)
        res = th.where(inputs >= 0, a / b * inputs, a * (th.exp(inputs / b) -
            1))
        return res


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
