import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
from typing import *


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        logits = logits.masked_fill(mask == 0, -1000000000.0)
    attention_map = F.softmax(logits, dim=-1)
    if dropout is not None:
        attention_map = dropout(attention_map)
    return torch.matmul(attention_map, value)


class MultiHeadAttention(nn.Module):

    def __init__(self, hidden_dim, n_heads, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // n_heads
        self.n_heads = n_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        k_project = self.k_proj(key)
        q_project = self.q_proj(query)
        v_project = self.v_proj(value)
        k_reshape = k_project.view(batch_size, -1, self.n_heads, self.head_dim
            ).transpose(1, 2)
        q_reshape = q_project.view(batch_size, -1, self.n_heads, self.head_dim
            ).transpose(1, 2)
        v_reshape = v_project.view(batch_size, -1, self.n_heads, self.head_dim
            ).transpose(1, 2)
        scores = attention(q_reshape, k_reshape, v_reshape, mask, self.dropout)
        scores = scores.transpose(1, 2).contiguous()
        scores = scores.view(batch_size, -1, self.hidden_dim)
        return self.output_proj(scores)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_dim': 4, 'n_heads': 4}]
