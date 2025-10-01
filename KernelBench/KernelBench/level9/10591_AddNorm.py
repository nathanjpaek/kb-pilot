import torch
from torch import nn


class AddNorm(nn.Module):

    def __init__(self, features, dropout=0.0, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(features)

    def forward(self, x, y):
        return self.ln(self.dropout(y) + x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'features': 4}]
