import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.0):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        self.d_k = d_model // h
        self.h = h
        self.W_q = nn.Linear(d_model, self.h * self.d_k)
        self.W_k = nn.Linear(d_model, self.h * self.d_k)
        self.W_v = nn.Linear(d_model, self.h * self.d_k)
        self.W_o = nn.Linear(self.h * self.d_k, d_model, bias=False)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def __attention(self, query, key, value, mask=None, dropout=None):
        """Compute 'Scaled Dot Product Attention'"""
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(1) == 0,
                -1000000000.0)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None):
        assert key.dim() == 3 and query.dim() == 3 and value.dim() == 3
        batch_size = query.size(0)
        query = self.W_q(query).view(batch_size, -1, self.h, self.d_k)
        key = self.W_k(key).view(batch_size, -1, self.h, self.d_k)
        value = self.W_v(value).view(batch_size, -1, self.h, self.d_k)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        x, self.attn = self.__attention(query, key, value, dropout=self.
            dropout, mask=mask)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h *
            self.d_k)
        x = self.W_o(x)
        return x


class ResidualSelfAttention0(nn.Module):
    """
    Residual connection and layer norm with self attention
    Permutation EQUIvariant
    """

    def __init__(self, heads, d_model, dropout=0.0):
        super(ResidualSelfAttention0, self).__init__()
        self.size = d_model
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn = MultiHeadedAttention(heads, d_model, dropout=dropout)
        self.sublayer = lambda x: self.attn(x, x, x)

    def forward(self, x):
        assert x.dim() == 3
        """Apply residual connection to any sublayer with the _same size_."""
        return x + self.norm(self.dropout(self.sublayer(x)))


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'heads': 4, 'd_model': 4}]
