import math
import torch
from torch import Tensor
from typing import Optional
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter


class LocalLinearCF(nn.Module):

    def __init__(self, in_ch: 'int', out_ch: 'int', n_freqs: 'int', bias:
        'bool'=True):
        super().__init__()
        self.n_freqs = n_freqs
        self.register_parameter('weight', Parameter(torch.zeros(in_ch,
            out_ch, n_freqs), requires_grad=True))
        if bias:
            self.bias: 'Optional[Tensor]'
            self.register_parameter('bias', Parameter(torch.zeros(out_ch, 1,
                n_freqs), requires_grad=True))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: 'Tensor') ->Tensor:
        x = torch.einsum('bctf,cof->botf', x, self.weight)
        if self.bias is not None:
            x = x + self.bias
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_ch': 4, 'out_ch': 4, 'n_freqs': 4}]
