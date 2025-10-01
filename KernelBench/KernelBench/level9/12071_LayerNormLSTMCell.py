import torch
import torch.distributed
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormLSTMCell(nn.LSTMCell):

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__(input_size, hidden_size, bias)
        self.ln_ih = nn.LayerNorm(4 * hidden_size)
        self.ln_hh = nn.LayerNorm(4 * hidden_size)
        self.ln_ho = nn.LayerNorm(hidden_size)

    def forward(self, input, hidden=None):
        if hidden is None:
            hx = input.new_zeros(input.size(0), self.hidden_size,
                requires_grad=False)
            cx = input.new_zeros(input.size(0), self.hidden_size,
                requires_grad=False)
        else:
            hx, cx = hidden
        gates = self.ln_ih(F.linear(input, self.weight_ih, self.bias_ih)
            ) + self.ln_hh(F.linear(hx, self.weight_hh, self.bias_hh))
        i, f, o = gates[:, :3 * self.hidden_size].sigmoid().chunk(3, 1)
        g = gates[:, 3 * self.hidden_size:].tanh()
        cy = f * cx + i * g
        hy = o * self.ln_ho(cy).tanh()
        return hy, cy


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4}]
