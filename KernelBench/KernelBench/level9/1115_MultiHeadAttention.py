import torch
import numpy as np
import torch.nn as nn


class SelfAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        key_dim = key.size(-1)
        attn = torch.matmul(query / np.sqrt(key_dim), key.transpose(2, 3))
        if mask is not None:
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask == 0, -1000000000.0)
        attn = self.dropout(torch.softmax(attn, dim=-1))
        output = torch.matmul(attn, value)
        return output


class MultiHeadAttention(nn.Module):

    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.self_attention = SelfAttention(dropout)
        self.num_heads = num_heads
        self.dim_per_head = embedding_dim // num_heads
        self.query_projection = nn.Linear(embedding_dim, embedding_dim)
        self.key_projection = nn.Linear(embedding_dim, embedding_dim)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query = self.query_projection(query)
        key = self.key_projection(key)
        value = self.value_projection(value)
        query = query.view(batch_size, -1, self.num_heads, self.dim_per_head
            ).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.dim_per_head
            ).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.dim_per_head
            ).transpose(1, 2)
        scores = self.self_attention(query, key, value, mask)
        output = scores.transpose(1, 2).contiguous().view(batch_size, -1,
            self.embedding_dim)
        output = self.out(output)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'embedding_dim': 4, 'num_heads': 4}]
