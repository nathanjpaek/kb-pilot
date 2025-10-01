import torch
import torch.nn.functional as F
from torch import nn


class SelfAttention(nn.Module):

    def __init__(self, embedding_dimension, num_heads):
        super().__init__()
        assert embedding_dimension % num_heads == 0, f'embedding dimension must be divisible by number of heads, got embedding_dimension={embedding_dimension!r}, num_heads={num_heads!r}'
        self.num_heads = num_heads
        k = embedding_dimension
        self.to_keys = nn.Linear(k, k * num_heads, bias=False)
        self.to_queries = nn.Linear(k, k * num_heads, bias=False)
        self.to_values = nn.Linear(k, k * num_heads, bias=False)
        self.unify_heads = nn.Linear(num_heads * k, k)

    def forward(self, x):
        b, t, k = x.size()
        h = self.num_heads
        keys = self.to_keys(x).view(b, t, h, k)
        queries = self.to_queries(x).view(b, t, h, k)
        values = self.to_values(x).view(b, t, h, k)
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, k)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
        values = values.transpose(1, 2).contiguous().view(b * h, t, k)
        dot = torch.bmm(queries, keys.transpose(1, 2))
        dot = dot / k ** (1 / 2)
        dot = F.softmax(dot, dim=2)
        out = torch.bmm(dot, values).view(b, h, t, k)
        out = out.transpose(1, 2).contiguous().view(b, t, h * k)
        return self.unify_heads(out)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'embedding_dimension': 4, 'num_heads': 4}]
