import math
import torch
from torch import nn


class MultiHeadedAttention(nn.Module):

    def __init__(self, dim, n_head, bias=True, dropout=0):
        super().__init__()
        self.dim_head = dim // n_head
        self.n_head = n_head
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(dim, dim)

    def forward(self, input):
        batch, length, dim = input.shape
        qkv = self.qkv(input).reshape(batch, length, 3, self.n_head, self.
            dim_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = q @ k.transpose(-2, -1) / math.sqrt(self.dim_head)
        attn = torch.softmax(attn, -1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).reshape(batch, length, dim)
        out = self.linear(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4, 'n_head': 4}]
