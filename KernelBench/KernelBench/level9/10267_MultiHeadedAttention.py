import math
import torch
from typing import Optional
import torch.nn.functional as F
from torch import nn


def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1000000000.0)
    weights = F.softmax(scores, dim=-1)
    if dropout is not None:
        weights = dropout(weights)
    out = torch.matmul(weights, value)
    return out, weights


class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_head = d_model // h
        self.h = h
        self.proj_q = nn.Linear(d_model, d_model)
        self.proj_k = nn.Linear(d_model, d_model)
        self.proj_v = nn.Linear(d_model, d_model)
        self.ret_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query: 'torch.Tensor', key: 'torch.Tensor', value:
        'torch.Tensor', mask: 'Optional[torch.Tensor]'=None):
        """
        query: (batch_size, seq_len, dmodel)
        key: (batch_size, seq_len, dmodel)
        value: (batch_size, seq_len, dmodel)
        mask: (batch_size, seq_len)
        """
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)
        seq_len = query.size(1)
        query = self.proj_q(query).view(batch_size, seq_len, self.h, self.
            d_head).transpose(1, 2)
        key = self.proj_k(key).view(batch_size, seq_len, self.h, self.d_head
            ).transpose(1, 2)
        value = self.proj_v(value).view(batch_size, seq_len, self.h, self.
            d_head).transpose(1, 2)
        x, _ = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.h *
            self.d_head)
        return self.ret_proj(x)


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {'h': 4, 'd_model': 4}]
