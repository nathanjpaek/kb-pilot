import torch
import torch.nn as nn


class SelfAttention(nn.Module):

    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dimension = embed_size // heads
        assert self.head_dimension * self.heads == self.embed_size, 'Embed size needs to be divisible by heads'
        self.values = nn.Linear(self.head_dimension, self.head_dimension,
            bias=False)
        self.keys = nn.Linear(self.head_dimension, self.head_dimension,
            bias=False)
        self.queries = nn.Linear(self.head_dimension, self.head_dimension,
            bias=False)
        self.fc_out = nn.Linear(self.head_dimension * self.heads, self.
            embed_size)

    def forward(self, queries, keys, values):
        N = queries.shape[0]
        key_len, query_len, value_len = keys.shape[1], queries.shape[1
            ], values.shape[1]
        values = values.reshape(N, value_len, self.heads, self.head_dimension)
        keys = keys.reshape(N, key_len, self.heads, self.head_dimension)
        queries = queries.reshape(N, query_len, self.heads, self.head_dimension
            )
        keys = self.keys(keys)
        values = self.values(values)
        queries = self.queries(queries)
        energy = torch.einsum('nqhd, nkhd-> nhqk', [queries, keys])
        attention = torch.softmax(energy / self.embed_size ** (1 / 2), dim=3)
        out = torch.einsum('nhql, nlhd -> nqhd', [attention, values]).reshape(N
            , query_len, self.head_dimension * self.heads)
        out = self.fc_out(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 1]), torch.rand([4, 4, 4, 1]), torch.rand(
        [4, 4, 4, 1])]


def get_init_inputs():
    return [[], {'embed_size': 4, 'heads': 4}]
