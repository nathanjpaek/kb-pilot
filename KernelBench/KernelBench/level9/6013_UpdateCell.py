import torch
from torch import nn
import torch as th


class UpdateCell(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.x2i = nn.Linear(input_dim, 2 * output_dim, bias=True)
        self.h2h = nn.Linear(output_dim, 2 * output_dim, bias=False)

    def forward(self, x, hidden):
        i_i, i_n = self.x2i(x).chunk(2, 1)
        h_i, h_n = self.h2h(hidden).chunk(2, 1)
        input_gate = th.sigmoid(i_i + h_i)
        new_gate = th.tanh(i_n + h_n)
        return new_gate + input_gate * (hidden - new_gate)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4}]
