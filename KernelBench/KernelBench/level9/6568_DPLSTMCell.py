import math
import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.parallel
from typing import Optional
from typing import Tuple


class LSTMLinear(nn.Linear):
    """
    This function is the same as a nn.Linear layer, except that in the backward pass
    the grad_samples get accumulated (instead of being concatenated as in the standard
    nn.Linear)
    """

    def __init__(self, in_features: 'int', out_features: 'int', bias:
        'bool'=True):
        super().__init__(in_features, out_features, bias)


class DPLSTMCell(nn.Module):
    """
    Internal-only class. Implements *one* step of LSTM so that a LSTM layer can be seen as repeated
    applications of this class.
    """

    def __init__(self, input_size: 'int', hidden_size: 'int', bias: 'bool'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.ih = LSTMLinear(input_size, 4 * hidden_size, bias=self.bias)
        self.hh = LSTMLinear(hidden_size, 4 * hidden_size, bias=self.bias)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets parameters by initializing them from an uniform distribution.
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def set_max_batch_length(self, max_batch_length: 'int') ->None:
        """
        Sets max batch length
        """
        self.ih.max_batch_len = max_batch_length
        self.hh.max_batch_len = max_batch_length

    def forward(self, x: 'torch.Tensor', h_prev: 'torch.Tensor', c_prev:
        'torch.Tensor', batch_size_t: 'Optional[int]'=None) ->Tuple[torch.
        Tensor, torch.Tensor]:
        if batch_size_t is None:
            gates = self.ih(x) + self.hh(h_prev)
        else:
            gates = self.ih(x) + self.hh(h_prev[:batch_size_t, :])
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
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4, 'bias': 4}]
