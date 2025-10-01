import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadAttention(nn.Module):

    def __init__(self, d_model, heads, k_dim=None, v_dim=None, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        if k_dim is None:
            k_dim = d_model
        if v_dim is None:
            v_dim = d_model
        self.heads = heads
        self.d_model = d_model
        self.d_k = d_model // heads
        self.to_query = nn.Linear(d_model, d_model)
        self.to_key = nn.Linear(k_dim, d_model)
        self.to_value = nn.Linear(v_dim, d_model)
        self.to_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        batch, L = query.shape[:2]
        q = self.to_query(query).view(batch, L, self.heads, self.d_k).permute(
            0, 2, 1, 3)
        k = self.to_key(key).view(batch, L, self.heads, self.d_k).permute(0,
            2, 1, 3)
        v = self.to_value(value).view(batch, L, self.heads, self.d_k).permute(
            0, 2, 1, 3)
        attention = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        out = torch.matmul(attention, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch, L, -1)
        out = self.to_out(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {'d_model': 4, 'heads': 4}]
