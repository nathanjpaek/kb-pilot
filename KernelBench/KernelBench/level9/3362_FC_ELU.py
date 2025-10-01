import torch
from torch import nn


class FC_ELU(nn.Module):

    def __init__(self, in_dim, hidden_units):
        super(FC_ELU, self).__init__()
        self.fc = nn.Linear(in_dim, hidden_units)
        self.elu = nn.ELU()

    def forward(self, x):
        out = self.fc(x)
        out = self.elu(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'hidden_units': 4}]
