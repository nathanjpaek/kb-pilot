import torch
import torch.nn as nn


class IA_gate(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(IA_gate, self).__init__()
        self.IA = nn.Linear(in_dim, out_dim)

    def forward(self, x, IA_head):
        a = self.IA(IA_head)
        a = 1.0 + torch.tanh(a)
        a = a.unsqueeze(-1).unsqueeze(-1)
        x = a * x
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4}]
