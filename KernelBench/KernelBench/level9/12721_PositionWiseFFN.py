import torch
from torch import nn
from torch.nn.functional import relu


class PositionWiseFFN(nn.Module):

    def __init__(self, model_dim, dropout=0.0):
        super().__init__()
        dff = model_dim * 4
        self.l = nn.Linear(model_dim, dff)
        self.o = nn.Linear(dff, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        o = relu(self.l(x))
        o = self.o(o)
        o = self.dropout(o)
        o = self.layer_norm(x + o)
        return o


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'model_dim': 4}]
