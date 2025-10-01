import torch
from abc import ABC
import torch.nn as nn


class PositionwiseFeedForward(nn.Module, ABC):

    def __init__(self, d_in, d_hidden, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hidden)
        self.w_2 = nn.Linear(d_hidden, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-06)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, x):
        residual = x
        x = self.w_2(self.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_in': 4, 'd_hidden': 4}]
