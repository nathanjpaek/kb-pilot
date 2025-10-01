import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import *


class AUGRUCell(nn.Module):
    """ Effect of GRU with attentional update gate (AUGRU)

        Reference:
        -  Deep Interest Evolution Network for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1809.03672, 2018.
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(AUGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = nn.Parameter(torch.Tensor(3 * hidden_size, input_size)
            )
        self.register_parameter('weight_ih', self.weight_ih)
        self.weight_hh = nn.Parameter(torch.Tensor(3 * hidden_size,
            hidden_size))
        self.register_parameter('weight_hh', self.weight_hh)
        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter('bias_ih', self.bias_ih)
            self.bias_hh = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter('bias_ih', self.bias_hh)
            for tensor in [self.bias_ih, self.bias_hh]:
                nn.init.zeros_(tensor)
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

    def forward(self, input, hx, att_score):
        gi = F.linear(input, self.weight_ih, self.bias_ih)
        gh = F.linear(hx, self.weight_hh, self.bias_hh)
        i_r, i_z, i_n = gi.chunk(3, 1)
        h_r, h_z, h_n = gh.chunk(3, 1)
        reset_gate = torch.sigmoid(i_r + h_r)
        update_gate = torch.sigmoid(i_z + h_z)
        new_state = torch.tanh(i_n + reset_gate * h_n)
        att_score = att_score.view(-1, 1)
        update_gate = att_score * update_gate
        hy = (1.0 - update_gate) * hx + update_gate * new_state
        return hy


def get_inputs():
    return [torch.rand([64, 4]), torch.rand([64, 4]), torch.rand([16, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4}]
