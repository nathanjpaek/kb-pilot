import torch
import torch.autograd
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, n_in, n_out, dropout=0):
        super().__init__()
        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_in': 4, 'n_out': 4}]
