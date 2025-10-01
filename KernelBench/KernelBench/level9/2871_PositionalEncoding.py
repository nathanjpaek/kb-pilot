import torch
import numpy as np
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, module_dim=512, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, module_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, module_dim, 2).float() * (-np.
            log(10000.0) / module_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = nn.Parameter(pe.unsqueeze(0).transpose(0, 1),
            requires_grad=False)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 512])]


def get_init_inputs():
    return [[], {}]
