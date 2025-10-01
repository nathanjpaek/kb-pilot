import torch
from torch import nn


class PositionalEncoding(nn.Module):

    def __init__(self, patch_num, d_model, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.pe = nn.Parameter(torch.rand(patch_num + 1, d_model))
        self.add_positional_encoding = lambda x: x + self.pe[:x.size(1)
            ].unsqueeze(0)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.add_positional_encoding(x)
        return self.dropout(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'patch_num': 4, 'd_model': 4}]
