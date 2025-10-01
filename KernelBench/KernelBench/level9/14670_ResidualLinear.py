import torch
import torch.nn as nn


class ResidualLinear(nn.Module):

    def __init__(self, hidden_dim, norm1=None, norm2=None):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = norm1
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = norm2
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        f = self.linear1(input)
        if self.norm1 is not None:
            f = self.norm1(f)
        f = self.relu(f)
        f = self.linear2(f)
        if self.norm2 is not None:
            f = self.norm2(f)
        f = f + input
        f = self.relu(f)
        return f


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_dim': 4}]
