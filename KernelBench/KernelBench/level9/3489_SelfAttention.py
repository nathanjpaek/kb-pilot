import torch
import torch.nn as nn


class SelfAttention(nn.Module):

    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert self.head_dim * self.heads == self.embed_size, 'Embed size need to be div by heads'
        self.value = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.key = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.query = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(self.heads * self.head_dim, self.embed_size)

    def forward(self, values, keys, queries, mask):
        N = queries.shape[0]
        values_len, keys_len, queries_len = values.shape[1], keys.shape[1
            ], queries.shape[1]
        values = values.reshape(N, values_len, self.heads, self.head_dim)
        keys = keys.reshape(N, keys_len, self.heads, self.head_dim)
        queries = queries.reshape(N, queries_len, self.heads, self.head_dim)
        values = self.value(values)
        keys = self.key(keys)
        queries = self.query(queries)
        energy = torch.einsum('NQHD, NKHD -> NHQK', [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float(1e-12))
        attention = torch.softmax(energy / self.head_dim ** 0.5, dim=3)
        out = torch.einsum('NVHD, NHQK->NQHD', [values, attention]).reshape(N,
            queries_len, self.heads * self.head_dim)
        out = self.fc_out(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 1]), torch.rand([4, 4, 4, 1]), torch.rand(
        [4, 4, 4, 1]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'embed_size': 4, 'heads': 4}]
