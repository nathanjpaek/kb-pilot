import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


def truncated_normal(t, mean=0.0, std=0.01):
    torch.nn.init.normal_(t, mean=mean, std=std)
    while True:
        cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
        if torch.sum(cond):
            t = torch.where(cond, torch.nn.init.normal_(torch.ones_like(t),
                mean=mean, std=std), t)
        else:
            break
    return t


class ScaleHardsigmoid(torch.nn.Module):
    """
    This is a scaled addition (x+1)/2.
    """

    def __init__(self, scale=3):
        super(ScaleHardsigmoid, self).__init__()
        self.scale = scale

    def forward(self, x) ->str:
        return torch.nn.Hardsigmoid()(x * self.scale)


class HardMGUCellNUA(torch.nn.Module):
    """
    This is a minimal gated unit by replacing sigmoid and tanh with scalehardsigmoid and hardtanh if hard is set to True.
    Batch is always the dim[0].
    Refer to "Simplified Minimal Gated Unit Variations for Recurrent Neural Networks" and "Gate-Variants of Gated Recurrent Unit (GRU) Neural Networks" for more details.
    Hardtanh is to bound data to the legal unary range.
    This module follows the uBrain implementation style to maximize hardware reuse.
    """

    def __init__(self, input_size: 'int', hidden_size: 'int', bias: 'bool'=
        True, hard: 'bool'=True) ->None:
        super(HardMGUCellNUA, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.hard = hard
        if hard is True:
            self.fg_sigmoid = ScaleHardsigmoid()
            self.ng_tanh = nn.Hardtanh()
        else:
            self.fg_sigmoid = nn.Sigmoid()
            self.ng_tanh = nn.Tanh()
        self.weight_f = nn.Parameter(torch.empty((hidden_size, hidden_size +
            input_size)))
        self.weight_n = nn.Parameter(torch.empty((hidden_size, hidden_size +
            input_size)))
        if bias:
            self.bias_f = nn.Parameter(torch.empty(hidden_size))
            self.bias_n = nn.Parameter(torch.empty(hidden_size))
        else:
            self.register_parameter('bias_f', None)
            self.register_parameter('bias_n', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data = truncated_normal(weight, mean=0, std=stdv)

    def forward(self, input: 'Tensor', hx: 'Optional[Tensor]'=None) ->Tensor:
        if hx is None:
            hx = torch.zeros(input.size()[0], self.hidden_size, dtype=input
                .dtype, device=input.device)
        self.fg_ug_in = torch.cat((hx, input), 1)
        self.fg_in = F.linear(self.fg_ug_in, self.weight_f, self.bias_f)
        self.fg = self.fg_sigmoid(self.fg_in)
        self.fg_hx = self.fg * hx
        self.ng_ug_in = torch.cat((self.fg_hx, input), 1)
        self.ng_in = F.linear(self.ng_ug_in, self.weight_n, self.bias_n)
        self.ng = self.ng_tanh(self.ng_in)
        self.fg_ng_inv = (0 - self.fg) * self.ng
        hy = self.ng + self.fg_ng_inv + self.fg_hx
        return hy


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4}]
