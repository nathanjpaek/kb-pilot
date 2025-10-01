import torch
import torch.nn as nn
import torch.nn.functional as F


class AUGRUCell(nn.Module):
    ' Effect of GRU with attentional update gate (AUGRU). AUGRU combines attention mechanism and GRU seamlessly.\n\n    Formally:\n        ..math: \tilde{{u}}_{t}^{\\prime}=a_{t} * {u}_{t}^{\\prime} \\\n                {h}_{t}^{\\prime}=\\left(1-\tilde{{u}}_{t}^{\\prime}\right) \\circ {h}_{t-1}^{\\prime}+\tilde{{u}}_{t}^{\\prime} \\circ \tilde{{h}}_{t}^{\\prime}\n\n    '

    def __init__(self, input_size, hidden_size, bias=True):
        super(AUGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = nn.Parameter(torch.randn(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(3 * hidden_size, hidden_size)
            )
        if bias:
            self.bias_ih = nn.Parameter(torch.zeros(3 * hidden_size))
            self.bias_hh = nn.Parameter(torch.zeros(3 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

    def forward(self, input, hidden_output, att_score):
        gi = F.linear(input, self.weight_ih, self.bias_ih)
        gh = F.linear(hidden_output, self.weight_hh, self.bias_hh)
        i_r, i_u, i_h = gi.chunk(3, 1)
        h_r, h_u, h_h = gh.chunk(3, 1)
        reset_gate = torch.sigmoid(i_r + h_r)
        update_gate = torch.sigmoid(i_u + h_u)
        new_state = torch.tanh(i_h + reset_gate * h_h)
        att_score = att_score.view(-1, 1)
        update_gate = att_score * update_gate
        hy = (1 - update_gate) * hidden_output + update_gate * new_state
        return hy


def get_inputs():
    return [torch.rand([64, 4]), torch.rand([64, 4]), torch.rand([16, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4}]
