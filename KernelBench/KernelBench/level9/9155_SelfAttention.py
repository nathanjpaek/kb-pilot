import torch
import torch.nn as nn
from torch.nn import functional as F


def mask_fn(x, mask_diagonal=False):
    _b, h, w = x.size()
    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    mask = torch.zeros_like(x)
    mask[:, indices[0], indices[1]] = 1
    final_mask = (mask == 1) & (x == 0)
    x.masked_fill_(final_mask, float('-inf'))
    return x


class SelfAttention(nn.Module):

    def __init__(self, emb, heads=1, mask=True):
        super().__init__()
        self.emb = emb
        self.heads = heads
        self.mask = mask
        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)
        self.unifyheads = nn.Linear(emb * heads, emb, bias=False)

    def forward(self, x):
        b, t, k = x.size()
        h = self.heads
        keys = self.tokeys(x).contiguous().view(b, t, h, k)
        queries = self.toqueries(x).contiguous().view(b, t, h, k)
        values = self.tovalues(x).contiguous().view(b, t, h, k)
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, k)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
        values = values.transpose(1, 2).contiguous().view(b * h, t, k)
        dot = torch.bmm(keys, queries.transpose(1, 2))
        if self.mask:
            dot = mask_fn(dot)
        dot = F.softmax(dot, dim=2)
        out = torch.bmm(dot, values).contiguous().view(b, t, h * k)
        out = self.unifyheads(out)
        assert out.size() == (b, t, k)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'emb': 4}]
