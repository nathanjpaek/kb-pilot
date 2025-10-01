import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.distributed


def matmul(x, y):
    if x.dim() == y.dim():
        return x @ y
    if x.dim() == y.dim() - 1:
        return (x.unsqueeze(-2) @ y).squeeze(-2)
    return (x @ y.unsqueeze(-2)).squeeze(-2)


class Linear(nn.Linear):

    def forward(self, x):
        size = x.size()
        return super().forward(x.contiguous().view(-1, size[-1])).view(*
            size[:-1], -1)


class Attention(nn.Module):

    def __init__(self, d_key, drop_ratio, causal):
        super().__init__()
        self.scale = math.sqrt(d_key)
        self.dropout = nn.Dropout(drop_ratio)
        self.causal = causal

    def forward(self, query, key, value):
        dot_products = matmul(query, key.transpose(1, 2))
        if query.dim() == 3 and (self is None or self.causal):
            tri = torch.ones(key.size(1), key.size(1)).triu(1) * INF
            if key.is_cuda:
                tri = tri
            dot_products.data.sub_(tri.unsqueeze(0))
        return matmul(self.dropout(F.softmax(dot_products / self.scale, dim
            =2)), value)


class MultiHead(nn.Module):

    def __init__(self, d_key, d_value, n_heads, drop_ratio, causal=False):
        super().__init__()
        self.attention = Attention(d_key, drop_ratio, causal=causal)
        self.wq = Linear(d_key, d_key, bias=False)
        self.wk = Linear(d_key, d_key, bias=False)
        self.wv = Linear(d_value, d_value, bias=False)
        self.wo = Linear(d_value, d_key, bias=False)
        self.n_heads = n_heads

    def forward(self, query, key, value):
        query, key, value = self.wq(query), self.wk(key), self.wv(value)
        query, key, value = (x.chunk(self.n_heads, -1) for x in (query, key,
            value))
        return self.wo(torch.cat([self.attention(q, k, v) for q, k, v in
            zip(query, key, value)], -1))


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {'d_key': 4, 'd_value': 4, 'n_heads': 4, 'drop_ratio': 0.5}]
