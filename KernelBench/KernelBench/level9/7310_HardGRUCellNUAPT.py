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


class HardGRUCellNUAPT(torch.nn.RNNCellBase):
    """
    This is a standard GRUCell by replacing sigmoid and tanh with scalehardsigmoid and hardtanh if hard is set to True.
    Batch is always the dim[0].
    This module is not fully unary computing aware (NUA), i.e., not all intermediate data are bounded to the legal unary range.
    This module follows the PyTorch implementation style (PT).
    """

    def __init__(self, input_size: 'int', hidden_size: 'int', bias: 'bool'=
        True, hard: 'bool'=True) ->None:
        super(HardGRUCellNUAPT, self).__init__(input_size, hidden_size,
            bias, num_chunks=3)
        self.hard = hard
        if hard is True:
            self.resetgate_sigmoid = ScaleHardsigmoid()
            self.updategate_sigmoid = ScaleHardsigmoid()
            self.newgate_tanh = nn.Hardtanh()
        else:
            self.resetgate_sigmoid = nn.Sigmoid()
            self.updategate_sigmoid = nn.Sigmoid()
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
        gate_i = F.linear(input, self.weight_ih, self.bias_ih)
        gate_h = F.linear(hx, self.weight_hh, self.bias_hh)
        i_r, i_z, i_n = gate_i.chunk(3, 1)
        h_r, h_z, h_n = gate_h.chunk(3, 1)
        resetgate_in = i_r + h_r
        updategate_in = i_z + h_z
        resetgate = self.resetgate_sigmoid(resetgate_in)
        updategate = self.updategate_sigmoid(updategate_in)
        newgate_in = i_n + resetgate * h_n
        newgate = self.newgate_tanh(newgate_in)
        hy = (1 - updategate) * newgate + updategate * hx
        return hy


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4}]
