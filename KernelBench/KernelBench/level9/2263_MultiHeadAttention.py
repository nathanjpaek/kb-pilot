import torch
import numpy as np
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention Layer"""

    def __init__(self, hidden_size, num_attention_heads, attention_dropout_prob
        ):
        super(MultiHeadAttention, self).__init__()
        self.h = num_attention_heads
        self.d_k = hidden_size // num_attention_heads
        self.w_q = nn.Linear(hidden_size, hidden_size)
        self.w_k = nn.Linear(hidden_size, hidden_size)
        self.w_v = nn.Linear(hidden_size, hidden_size)
        self.w_o = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(attention_dropout_prob)

    def forward(self, query, key, value, mask=None):
        batch_size, hidden_size = query.shape[0], query.shape[2]
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)
        q = q.view(batch_size, -1, self.h, self.d_k).permute(0, 2, 1, 3)
        k = k.view(batch_size, -1, self.h, self.d_k).permute(0, 2, 3, 1)
        v = v.view(batch_size, -1, self.h, self.d_k).permute(0, 2, 1, 3)
        attention_scores = torch.matmul(q, k) / np.sqrt(self.d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -10000.0
                )
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        y = torch.matmul(attention_probs, v)
        y = y.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, hidden_size
            )
        return self.w_o(y)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4, 'num_attention_heads': 4,
        'attention_dropout_prob': 0.5}]
