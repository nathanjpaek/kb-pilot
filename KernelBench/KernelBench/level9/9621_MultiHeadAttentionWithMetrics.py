import torch
import numpy as np
import torch.nn as nn


class MultiHeadAttentionWithMetrics(nn.Module):

    def __init__(self, ctx, heads_count, d_model, dropout_prob=0.1, mode=
        'self-attention'):
        super(MultiHeadAttentionWithMetrics, self).__init__()
        assert d_model % heads_count == 0
        assert mode in ('self-attention', 'memory-attention')
        self.context = ctx
        self.d_head = d_model // heads_count
        self.heads_count = heads_count
        self.mode = mode
        self.query_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.key_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.value_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.final_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.dropout = nn.Dropout(dropout_prob)
        self.softmax = nn.Softmax(dim=3)
        self.attention = None
        self.key_projected = None
        self.value_projected = None

    def forward(self, query, key, value, mask=None, layer_cache=None):
        batch_size, query_len, d_model = query.size()
        d_head = d_model // self.heads_count
        query_projected = self.query_projection(query)
        if layer_cache is None or layer_cache[self.mode] is None:
            key_projected = self.key_projection(key)
            value_projected = self.value_projection(value)
        elif self.mode == 'self-attention':
            key_projected = self.key_projection(key)
            value_projected = self.value_projection(value)
            key_projected = torch.cat([key_projected, layer_cache[self.mode
                ]['key_projected']], dim=1)
            value_projected = torch.cat([value_projected, layer_cache[self.
                mode]['value_projected']], dim=1)
        elif self.mode == 'memory-attention':
            key_projected = layer_cache[self.mode]['key_projected']
            value_projected = layer_cache[self.mode]['value_projected']
        self.key_projected = key_projected
        self.value_projected = value_projected
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
        self.attention = self.softmax(attention_weights)
        attention_dropped = self.dropout(self.attention)
        context_heads = torch.matmul(attention_dropped, value_heads)
        context_sequence = context_heads.transpose(1, 2).contiguous()
        context = context_sequence.view(batch_size, query_len, d_model)
        final_output = self.final_projection(context)
        return final_output

    def scaled_dot_product(self, query_heads, key_heads):
        """

        Args:
             query_heads: (batch_size, heads_count, query_len, d_head)
             key_heads: (batch_size, heads_count, key_len, d_head)
        """
        key_heads_transposed = key_heads.transpose(2, 3)
        dot_product = torch.matmul(query_heads, key_heads_transposed)
        attention_weights = dot_product / np.sqrt(self.d_head)
        return attention_weights


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {'ctx': 4, 'heads_count': 4, 'd_model': 4}]
