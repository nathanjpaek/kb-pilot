import math
import torch
from torch import nn
import torch.nn.functional as F


class SelfAttention_naive(nn.Module):

    def __init__(self, dim_emb, dim_internal, heads=8, mask=False, dropout=
        0.0, dtype=torch.float32):
        """
        A single self attention block

        :param dim_emb: embedding dimension
        :param dim_internal: dimension of internal representation, usually the same as dim_emb
        :param head: number of multi head
        :param mask

        """
        super().__init__()
        self.dim_emb = dim_emb
        self.dim_internal = dim_internal
        self.heads = heads
        self.mask = mask
        self.toqueries = nn.Linear(dim_emb, dim_internal).type(dtype)
        self.tokeys = nn.Linear(dim_emb, dim_internal).type(dtype)
        self.tovalues = nn.Linear(dim_emb, dim_internal).type(dtype)
        self.kSqrt_dim_emb = math.sqrt(self.dim_emb)

    def forward(self, x):
        _b, _t, e = x.size()
        assert e == self.dim_emb, f'Input embedding ({e}) should match the layer embedding ({self.dim_emb})'
        queries = self.toqueries(x)
        keys = self.tokeys(x)
        values = self.tovalues(x)
        keys_transposed = keys.transpose(-2, -1)
        dot = torch.matmul(queries, keys_transposed) / self.kSqrt_dim_emb
        p_attn = F.softmax(dot, dim=2)
        z = torch.matmul(p_attn, values)
        return z


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_emb': 4, 'dim_internal': 4}]
