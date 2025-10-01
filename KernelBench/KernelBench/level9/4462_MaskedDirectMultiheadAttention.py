import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedDirectMultiheadAttention(nn.Module):

    def __init__(self, d_in, d_out, heads, d_k=32, dropout=0.1):
        super(MaskedDirectMultiheadAttention, self).__init__()
        self.heads = heads
        self.scaling = 1 / math.sqrt(d_k)
        self.to_query = nn.Linear(d_in, heads * d_k)
        self.to_key = nn.Linear(d_in, heads * d_k)
        self.to_value = nn.Linear(d_out, d_out)
        self.to_out = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, query, key, value, mask):
        batch, N, L = value.shape[:3]
        q = self.to_query(query).view(batch, L, self.heads, -1).permute(0, 
            2, 1, 3)
        k = self.to_key(key).view(batch, L, self.heads, -1).permute(0, 2, 1, 3)
        v = self.to_value(value).view(batch, N, L, self.heads, -1).permute(
            0, 3, 1, 2, 4)
        q = q * self.scaling
        attention = torch.matmul(q, k.transpose(-2, -1))
        attention = attention.masked_fill(mask < 0.5, torch.finfo(q.dtype).min)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        out = torch.einsum('bhij,bhnjk->bhnik', attention, v)
        out = out.permute(0, 2, 3, 1, 4).contiguous().view(batch, N, L, -1)
        out = self.to_out(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_in': 4, 'd_out': 4, 'heads': 4}]
