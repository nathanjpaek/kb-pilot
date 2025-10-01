import torch
import numpy as np
from torch import nn


class SinActivation(nn.Module):

    def __init__(self):
        super(SinActivation, self).__init__()

    def forward(self, x):
        return torch.sin(x)


class CoordFC(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)
        nn.init.uniform_(self.layer.weight, -np.sqrt(9 / input_dim), np.
            sqrt(9 / input_dim))
        self.act = SinActivation()
        pass

    def forward(self, x):
        x = self.layer(x)
        out = self.act(x)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'hidden_dim': 4}]
