import torch
import numpy as np
import torch.nn as nn


class MatrixVectorScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        """
        q: tensor of shape (*, m, d_k)
        k: tensor of shape (*, l, d_k)
        v: tensor of shape (*, l, d_v)
        mask: None or tensor of shape (*, m, l) or (*, 1, l)

        returns: tensor of shape (*, m, d_v), tensor of shape(*, m, l)
        """
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = attn / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        return output, attn


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'temperature': 4}]
