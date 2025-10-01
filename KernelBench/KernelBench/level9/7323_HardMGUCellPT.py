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


class HardMGUCellPT(torch.nn.RNNCellBase):
    """
    This is a minimal gated unit by replacing sigmoid and tanh with scalehardsigmoid and hardtanh if hard is set to True.
    Batch is always the dim[0].
    Refer to "Simplified Minimal Gated Unit Variations for Recurrent Neural Networks" and "Gate-Variants of Gated Recurrent Unit (GRU) Neural Networks" for more details.
    Hardtanh is to bound data to the legal unary range.
    This module is fully unary computing aware, i.e., all intermediate data are bounded to the legal unary range.
    This module follows the PyTorch implementation style (PT).
    """

    def __init__(self, input_size: 'int', hidden_size: 'int', bias: 'bool'=
        True, hard: 'bool'=True) ->None:
        super(HardMGUCellPT, self).__init__(input_size, hidden_size, bias,
            num_chunks=2)
        self.hard = hard
        if hard is True:
            self.forgetgate_sigmoid = ScaleHardsigmoid()
            self.newgate_tanh = nn.Hardtanh()
        else:
            self.forgetgate_sigmoid = nn.Sigmoid()
            self.newgate_tanh = nn.Tanh()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data = truncated_normal(weight, mean=0, std=stdv)

    def forward(self, input: 'Tensor', hx: 'Optional[Tensor]'=None) ->Tensor:
        if hx is None:
            hx = torch.zeros(input.size()[0], self.hidden_size, dtype=input
                .dtype, device=input.device)
        self.gate_i = nn.Hardtanh()(F.linear(input, self.weight_ih, self.
            bias_ih))
        self.gate_h = nn.Hardtanh()(F.linear(hx, self.weight_hh, self.bias_hh))
        self.i_f, self.i_n = self.gate_i.chunk(2, 1)
        self.h_f, self.h_n = self.gate_h.chunk(2, 1)
        self.forgetgate_in = nn.Hardtanh()(self.i_f + self.h_f)
        self.forgetgate = self.forgetgate_sigmoid(self.forgetgate_in)
        self.newgate_prod = self.forgetgate * self.h_n
        self.newgate = self.newgate_tanh(self.i_n + self.newgate_prod)
        self.forgetgate_inv_prod = (0 - self.forgetgate) * self.newgate
        self.forgetgate_prod = self.forgetgate * hx
        hy = nn.Hardtanh()(self.newgate + self.forgetgate_inv_prod + self.
            forgetgate_prod)
        return hy


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4}]
