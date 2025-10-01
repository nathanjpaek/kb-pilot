from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn


class MaskLSTMCell(nn.Module):

    def __init__(self, options):
        super(MaskLSTMCell, self).__init__()
        self.n_in = options['n_in']
        self.n_out = options['n_out']
        self.input = nn.Linear(self.n_in, self.n_out * 4)
        self.hidden = nn.Linear(self.n_out, self.n_out * 4, bias=False)

    def forward(self, input, mask, h_prev, c_prev):
        activation = self.input(input) + self.hidden(h_prev)
        activation_i = activation[:, :self.n_out]
        activation_f = activation[:, self.n_out:self.n_out * 2]
        activation_c = activation[:, self.n_out * 2:self.n_out * 3]
        activation_o = activation[:, self.n_out * 3:self.n_out * 4]
        i = torch.sigmoid(activation_i)
        f = torch.sigmoid(activation_f)
        o = torch.sigmoid(activation_o)
        c = f * c_prev + i * torch.tanh(activation_c)
        c = mask[:, None] * c + (1 - mask)[:, None] * c_prev
        h = o * torch.tanh(c)
        h = mask[:, None] * h + (1 - mask)[:, None] * h_prev
        return h, c


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4]),
        torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'options': _mock_config(n_in=4, n_out=4)}]
