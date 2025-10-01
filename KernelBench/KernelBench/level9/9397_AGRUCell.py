import torch
import torch.nn as nn
import torch.nn.functional as F


class AGRUCell(nn.Module):
    ' Attention based GRU (AGRU). AGRU uses the attention score to replace the update gate of GRU, and changes the\n    hidden state directly.\n\n    Formally:\n        ..math: {h}_{t}^{\\prime}=\\left(1-a_{t}\right) * {h}_{t-1}^{\\prime}+a_{t} * \tilde{{h}}_{t}^{\\prime}\n\n        :math:`{h}_{t}^{\\prime}`, :math:`h_{t-1}^{\\prime}`, :math:`{h}_{t-1}^{\\prime}`,\n        :math: `\tilde{{h}}_{t}^{\\prime}` are the hidden state of AGRU\n\n    '

    def __init__(self, input_size, hidden_size, bias=True):
        super(AGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = nn.Parameter(torch.randn(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(3 * hidden_size, hidden_size)
            )
        if self.bias:
            self.bias_ih = nn.Parameter(torch.zeros(3 * hidden_size))
            self.bias_hh = nn.Parameter(torch.zeros(3 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

    def forward(self, input, hidden_output, att_score):
        gi = F.linear(input, self.weight_ih, self.bias_ih)
        gh = F.linear(hidden_output, self.weight_hh, self.bias_hh)
        i_r, _i_u, i_h = gi.chunk(3, 1)
        h_r, _h_u, h_h = gh.chunk(3, 1)
        reset_gate = torch.sigmoid(i_r + h_r)
        new_state = torch.tanh(i_h + reset_gate * h_h)
        att_score = att_score.view(-1, 1)
        hy = (1 - att_score) * hidden_output + att_score * new_state
        return hy


def get_inputs():
    return [torch.rand([16, 4]), torch.rand([16, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4}]
