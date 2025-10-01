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


class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""

    def __init__(self, features: 'int', eps: 'float'=1e-06):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: 'torch.Tensor'):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


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


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Block(nn.Module):
    """A standard Decoder block for GPT."""

    def __init__(self, d_model: 'int', n_heads: 'int', dropout: 'float'=0.1):
        super(Block, self).__init__()
        self.d_model = d_model
        self.d_inner = 4 * self.d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.layer_norm1 = LayerNorm(self.d_model)
        self.layer_norm2 = LayerNorm(self.d_model)
        self.multi_head_attn = MultiHeadedAttention(self.n_heads, self.
            d_model, self.dropout)
        self.feed_fwd = PositionwiseFeedForward(d_model, self.d_inner, self
            .dropout)

    def forward(self, x: 'torch.Tensor', mask: 'Optional[torch.Tensor]'=None):
        x = self.layer_norm1(x)
        attn_out = self.multi_head_attn(x, x, x, mask)
        res_1 = attn_out + x
        feed_fwd_out = self.feed_fwd(self.layer_norm2(res_1))
        out = res_1 + feed_fwd_out
        return out


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'n_heads': 4}]
