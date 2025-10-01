import math
import torch
import torch.nn as nn


class MultiHeadDense(nn.Module):

    def __init__(self, d):
        super(MultiHeadDense, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(d, d))
        self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) ->None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        b, _wh, _d = x.size()
        x = torch.bmm(x, self.weight.repeat(b, 1, 1))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'d': 4}]
