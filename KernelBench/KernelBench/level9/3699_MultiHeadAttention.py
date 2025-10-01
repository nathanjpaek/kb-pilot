import torch
import numpy as np
from torch import nn
import torch.nn.parallel


class MultiHeadAttention(nn.Module):

    def __init__(self, heads_count, d_model, dropout_prob):
        super().__init__()
        assert d_model % heads_count == 0, f'model dim {d_model} not divisible by {heads_count} heads'
        self.d_head = d_model // heads_count
        self.heads_count = heads_count
        self.query_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.key_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.value_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.final_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.dropout = nn.Dropout(dropout_prob)
        self.softmax = nn.Softmax(dim=3)
        self.attention = None

    def forward(self, query, key, value, mask=None):
        batch_size, query_len, d_model = query.size()
        d_head = d_model // self.heads_count
        query_projected = self.query_projection(query)
        key_projected = self.key_projection(key)
        value_projected = self.value_projection(value)
        batch_size, key_len, d_model = key_projected.size()
        batch_size, value_len, d_model = value_projected.size()
        query_heads = query_projected.view(batch_size, query_len, self.
            heads_count, d_head).transpose(1, 2)
        key_heads = key_projected.view(batch_size, key_len, self.
            heads_count, d_head).transpose(1, 2)
        value_heads = value_projected.view(batch_size, value_len, self.
            heads_count, d_head).transpose(1, 2)
        attention_weights = self.scaled_dot_product(query_heads, key_heads)
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).expand_as(attention_weights)
            attention_weights = attention_weights.masked_fill(mask_expanded,
                -1e+18)
        attention = self.softmax(attention_weights)
        attention_dropped = self.dropout(attention)
        context_heads = torch.matmul(attention_dropped, value_heads)
        context_sequence = context_heads.transpose(1, 2)
        context = context_sequence.reshape(batch_size, query_len, d_model)
        final_output = self.final_projection(context)
        return final_output

    def scaled_dot_product(self, query_heads, key_heads):
        key_heads_transposed = key_heads.transpose(2, 3)
        dot_product = torch.matmul(query_heads, key_heads_transposed)
        attention_weights = dot_product / np.sqrt(self.d_head)
        return attention_weights


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {'heads_count': 4, 'd_model': 4, 'dropout_prob': 0.5}]
