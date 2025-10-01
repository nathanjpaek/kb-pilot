import math
import torch
from torch import Tensor
from torch import nn
import torch.utils.data
import torch.utils.data.distributed
from typing import Optional


class RNNLinear(nn.Linear):
    """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    This module is the same as a ``torch.nn.Linear``` layer, except that in the backward pass
    the grad_samples get accumulated (instead of being concatenated as in the standard
    nn.Linear).

    When used with `PackedSequence`s, additional attribute `max_batch_len` is defined to determine
    the size of per-sample grad tensor.
    """
    max_batch_len: 'int'

    def __init__(self, in_features: 'int', out_features: 'int', bias:
        'bool'=True):
        super().__init__(in_features, out_features, bias)


class DPRNNCellBase(nn.Module):
    has_cell_state: 'bool' = False

    def __init__(self, input_size: 'int', hidden_size: 'int', bias: 'bool',
        num_chunks: 'int') ->None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.ih = RNNLinear(input_size, num_chunks * hidden_size, bias)
        self.hh = RNNLinear(hidden_size, num_chunks * hidden_size, bias)
        self.reset_parameters()

    def reset_parameters(self) ->None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def set_max_batch_length(self, max_batch_length: 'int') ->None:
        self.ih.max_batch_len = max_batch_length
        self.hh.max_batch_len = max_batch_length


class DPRNNCell(DPRNNCellBase):
    """An Elman RNN cell with tanh or ReLU non-linearity.

    DP-friendly drop-in replacement of the ``torch.nn.RNNCell`` module to use in ``DPRNN``.
    Refer to ``torch.nn.RNNCell`` documentation for the model description, parameters and inputs/outputs.
    """

    def __init__(self, input_size: 'int', hidden_size: 'int', bias: 'bool',
        nonlinearity: 'str'='tanh') ->None:
        super().__init__(input_size, hidden_size, bias, num_chunks=1)
        if nonlinearity not in ('tanh', 'relu'):
            raise ValueError(f'Unsupported nonlinearity: {nonlinearity}')
        self.nonlinearity = nonlinearity

    def forward(self, input: 'Tensor', hx: 'Optional[Tensor]'=None,
        batch_size_t: 'Optional[int]'=None) ->Tensor:
        if hx is None:
            hx = torch.zeros(input.shape[0], self.hidden_size, dtype=input.
                dtype, device=input.device)
        h_prev = hx
        gates = self.ih(input) + self.hh(h_prev if batch_size_t is None else
            h_prev[:batch_size_t, :])
        if self.nonlinearity == 'tanh':
            h_t = torch.tanh(gates)
        elif self.nonlinearity == 'relu':
            h_t = torch.relu(gates)
        else:
            raise RuntimeError(f'Unknown nonlinearity: {self.nonlinearity}')
        return h_t


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4, 'bias': 4}]
