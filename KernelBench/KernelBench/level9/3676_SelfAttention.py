import torch
from torch.nn import functional as F
from torch import nn


class SelfAttention(nn.Module):

    def __init__(self, k, heads=8):
        super().__init__()
        self.k, self.heads = k, heads
        self.toKeys = nn.Linear(k, k * heads, bias=False)
        self.toQueries = nn.Linear(k, k * heads, bias=False)
        self.toValues = nn.Linear(k, k * heads, bias=False)
        self.unifyHeads = nn.Linear(heads * k, k)

    def forward(self, x):
        b, t, k = x.size()
        h = self.heads
        queries = self.toQueries(x).view(b, t, h, k)
        keys = self.toKeys(x).view(b, t, h, k)
        values = self.toValues(x).view(b, t, h, k)
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, k)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
        values = values.transpose(1, 2).contiguous().view(b * h, t, k)
        queries = queries / k ** (1 / 4)
        keys = keys / k ** (1 / 4)
        dot = torch.bmm(queries, keys.transpose(1, 2))
        dot = F.softmax(dot, dim=2)
        out = torch.bmm(dot, values).view(b, h, t, k)
        out = out.transpose(1, 2).contiguous().view(b, t, h * k)
        return self.unifyHeads(out)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'k': 4}]
