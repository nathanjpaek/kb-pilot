import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn


class MultiHeadAttention(nn.Module):

    def __init__(self, embed_size, num_heads, dropout=0.2, batch_dim=0):
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
        if self.batch_dim == 0:
            out = self.batch_0(q, k, v, mask)
        elif self.batch_dim == 1:
            out = self.batch_1(q, k, v, mask)
        return out

    def batch_0(self, q, k, v, mask=None):
        q_batch_size, q_seq_len, _q_embed_size = q.size()
        k_batch_size, k_seq_len, _k_embed_size = k.size()
        v_batch_size, v_seq_len, _v_embed_size = v.size()
        q = self.Q(q).reshape(q_batch_size, q_seq_len, self.num_heads, self
            .head_size)
        k = self.K(k).reshape(k_batch_size, k_seq_len, self.num_heads, self
            .head_size)
        v = self.V(v).reshape(v_batch_size, v_seq_len, self.num_heads, self
            .head_size)
        scores = self.attention(q, k, v, self.num_heads, mask)
        concatenated = scores.reshape(v_batch_size, -1, self.embed_size)
        out = self.linear(concatenated)
        return out

    def batch_1(self, q, k, v, mask=None):
        q_seq_len, q_batch_size, _q_embed_size = q.size()
        k_seq_len, k_batch_size, _k_embed_size = k.size()
        v_seq_len, v_batch_size, _v_embed_size = v.size()
        q = self.Q(q).reshape(q_batch_size, q_seq_len, self.num_heads, self
            .head_size)
        k = self.K(k).reshape(k_batch_size, k_seq_len, self.num_heads, self
            .head_size)
        v = self.V(v).reshape(v_batch_size, v_seq_len, self.num_heads, self
            .head_size)
        scores = self.attention(q, k, v, self.num_heads, mask)
        concatenated = scores.reshape(-1, v_batch_size, self.embed_size)
        out = self.linear(concatenated)
        return out

    def attention(self, q, k, v, d_k, mask=None):
        scores = torch.einsum('bqhe,bkhe->bhqk', [q, k]) / math.sqrt(d_k)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout_layer(scores)
        out = torch.einsum('bhql,blhd->bqhd', [scores, v])
        return out


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {'embed_size': 4, 'num_heads': 4}]
