import torch
import torch.nn as nn


class Attention(nn.Module):
    """Attention mechanism written by Gustavo Aguilar https://github.com/gaguilar"""

    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.da = hidden_size
        self.dh = hidden_size
        self.W = nn.Linear(self.dh, self.da)
        self.v = nn.Linear(self.da, 1)

    def forward(self, inputs, mask):
        u = self.v(torch.tanh(self.W(inputs)))
        u = u.exp()
        u = mask.unsqueeze(2).float() * u
        sums = torch.sum(u, dim=1, keepdim=True)
        a = u / sums
        z = inputs * a
        return z, a.view(inputs.size(0), inputs.size(1))


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]
