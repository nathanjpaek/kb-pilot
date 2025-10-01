import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):

    def __init__(self, embedding_size, number_of_heads):
        super(MultiHeadAttention, self).__init__()
        self.embedding_size = embedding_size
        self.number_of_heads = number_of_heads
        self.head_dimension = embedding_size // number_of_heads
        assert self.head_dimension * number_of_heads == embedding_size, 'Embedding size needs to be divisible by the number of heads'
        self.value = nn.Linear(self.head_dimension, self.head_dimension,
            bias=False)
        self.key = nn.Linear(self.head_dimension, self.head_dimension, bias
            =False)
        self.query = nn.Linear(self.head_dimension, self.head_dimension,
            bias=False)
        self.full_connection = nn.Linear(number_of_heads * self.
            head_dimension, embedding_size)

    def forward(self, query, value, key, mask):
        batch_size = query.shape[0]
        value_len, key_len, query_len = value.shape[1], key.shape[1
            ], query.shape[1]
        values = value.reshape(batch_size, value_len, self.number_of_heads,
            self.head_dimension)
        keys = key.reshape(batch_size, key_len, self.number_of_heads, self.
            head_dimension)
        queries = query.reshape(batch_size, query_len, self.number_of_heads,
            self.head_dimension)
        values = self.value(values)
        keys = self.key(keys)
        queries = self.query(queries)
        attention = torch.einsum('bqhd,bkhd->bhqk', [queries, keys])
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float('-1e20'))
        attention /= self.embedding_size ** 0.5
        attention = torch.softmax(attention, dim=3)
        out = torch.einsum('bhql,blhd->bqhd', [attention, values]).reshape(
            batch_size, query_len, self.number_of_heads * self.head_dimension)
        out = self.full_connection(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 1]), torch.rand([4, 4, 4, 1]), torch.rand(
        [4, 4, 4, 1]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'embedding_size': 4, 'number_of_heads': 4}]
