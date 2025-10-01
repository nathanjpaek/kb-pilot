import torch
from torch import nn
import torch.nn.functional as F


class FeedForward(nn.Module):

    def __init__(self, dim, d_ff=128, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(dim, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, dim)

    def forward(self, x):
        x = self.dropout(F.gelu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
