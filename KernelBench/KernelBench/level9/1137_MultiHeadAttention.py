import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-headed Attention for input Query, Key, Value
    Multi-headed Attention is a module for attention mechanisms which runs through attention in several times in
    parallel, then the multiple outputs are concatenated and linearly transformed

    Args:
        embed_size  (int): Max embedding size
        num_heads   (int): Number of heads in multi-headed attention; Number of splits in the embedding size
        dropout     (float, optional): Percentage of Dropout to be applied in range 0 <= dropout <=1
        batch_dim   (int, optional): The dimension in which batch dimensions is
    """

    def __init__(self, embed_size: 'int', num_heads: 'int', dropout:
        'float'=0.2, batch_dim: 'int'=0):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_dim = batch_dim
        self.dropout_layer = nn.Dropout(dropout)
        self.head_size = self.embed_size // self.num_heads
        assert self.head_size * self.num_heads == self.embed_size, 'Heads cannot split Embedding size equally'
        self.Q = nn.Linear(self.embed_size, self.embed_size)
        self.K = nn.Linear(self.embed_size, self.embed_size)
        self.V = nn.Linear(self.embed_size, self.embed_size)
        self.linear = nn.Linear(self.embed_size, self.embed_size)

    def forward(self, q, k, v, mask=None):
        q_batch_size, q_seq_len, _q_embed_size = q.size()
        k_batch_size, k_seq_len, _k_embed_size = k.size()
        v_batch_size, v_seq_len, _v_embed_size = v.size()
        q = self.Q(q).reshape(q_batch_size, q_seq_len, self.num_heads, self
            .head_size)
        k = self.K(k).reshape(k_batch_size, k_seq_len, self.num_heads, self
            .head_size)
        v = self.V(v).reshape(v_batch_size, v_seq_len, self.num_heads, self
            .head_size)
        attention = self.attention(q, k, v, mask=mask)
        concatenated = attention.reshape(v_batch_size, -1, self.embed_size)
        out = self.linear(concatenated)
        return out

    def attention(self, q, k, v, mask=None):
        scores = torch.einsum('bqhe,bkhe->bhqk', [q, k])
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1000000000.0)
        scores /= math.sqrt(self.embed_size)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout_layer(scores)
        attention = torch.einsum('bhql,blhd->bqhd', [scores, v])
        return attention


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {'embed_size': 4, 'num_heads': 4}]
