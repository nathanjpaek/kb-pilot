import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.parallel
from typing import Optional
from typing import Tuple


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


class DPLSTMCell(DPRNNCellBase):
    """A long short-term memory (LSTM) cell.

    DP-friendly drop-in replacement of the ``torch.nn.LSTMCell`` module to use in ``DPLSTM``.
    Refer to ``torch.nn.LSTMCell`` documentation for the model description, parameters and inputs/outputs.
    """
    has_cell_state = True

    def __init__(self, input_size: 'int', hidden_size: 'int', bias: 'bool'
        ) ->None:
        super().__init__(input_size, hidden_size, bias, num_chunks=4)

    def forward(self, input: 'Tensor', hx:
        'Optional[Tuple[Tensor, Tensor]]'=None, batch_size_t:
        'Optional[int]'=None) ->Tuple[Tensor, Tensor]:
        if hx is None:
            zeros = torch.zeros(input.shape[0], self.hidden_size, dtype=
                input.dtype, device=input.device)
            hx = zeros, zeros
        h_prev, c_prev = hx
        if batch_size_t is None:
            gates = self.ih(input) + self.hh(h_prev)
        else:
            gates = self.ih(input) + self.hh(h_prev[:batch_size_t, :])
        i_t_input, f_t_input, g_t_input, o_t_input = torch.split(gates,
            self.hidden_size, 1)
        i_t = torch.sigmoid(i_t_input)
        f_t = torch.sigmoid(f_t_input)
        g_t = torch.tanh(g_t_input)
        o_t = torch.sigmoid(o_t_input)
        if batch_size_t is None:
            c_t = f_t * c_prev + i_t * g_t
        else:
            c_t = f_t * c_prev[:batch_size_t, :] + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4, 'bias': 4}]
