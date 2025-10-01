import torch
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):

    def __init__(self, model_dim, n_heads=1):
        super(Attention, self).__init__()
        self.model_dim = model_dim
        self.dim_per_head = model_dim // n_heads
        self.n_heads = n_heads
        self.fcq = nn.Linear(model_dim, self.dim_per_head * n_heads)
        self.fck = nn.Linear(model_dim, self.dim_per_head * n_heads)
        self.fcv = nn.Linear(model_dim, self.dim_per_head * n_heads)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, queries, keys, values):
        """
            queries: batch * model_dim
            keys: batch * ? * model_dim
            values: batch * ? * model_dim
        """
        residual = queries
        batch_size = queries.size(0)
        q = self.fcq(queries).view(batch_size * self.n_heads, 1, self.
            dim_per_head)
        k = self.fck(keys).view(batch_size, -1, self.n_heads, self.dim_per_head
            ).transpose(1, 2).reshape(batch_size * self.n_heads, -1, self.
            dim_per_head)
        v = self.fcv(values).view(batch_size, -1, self.n_heads, self.
            dim_per_head).transpose(1, 2).reshape(batch_size * self.n_heads,
            -1, self.dim_per_head)
        weight = th.bmm(q, k.transpose(1, 2)) / np.sqrt(self.dim_per_head)
        attn = th.bmm(F.softmax(weight, dim=-1), v)
        attn = attn.view(batch_size, self.n_heads * self.dim_per_head)
        return self.layer_norm(attn + residual)


def get_inputs():
    return [torch.rand([4, 1, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4,
        4, 4, 4])]


def get_init_inputs():
    return [[], {'model_dim': 4}]
