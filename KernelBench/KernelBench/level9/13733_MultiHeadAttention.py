import torch
import numpy as np
import torch.nn as nn


def dot_product_attention(queries, keys, values, normalise=True):
    """
    :param queries:[batch_size, N_target, key_size]
    :param keys:[batch_size, N_context, key_size]
    :param values: []
    :param normalise:
    :return:
    """
    key_size = keys.shape[-1]
    scale = np.sqrt(key_size)
    unnorm_weights = torch.matmul(queries, keys.transpose(-2, -1)) / scale
    if normalise:
        attention = torch.softmax(unnorm_weights, dim=-1)
    else:
        attention = torch.sigmoid(unnorm_weights)
    output = torch.matmul(attention, values)
    return output


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention class
    """

    def __init__(self, key_size, value_size, num_heads, key_hidden_size,
        normalise=True):
        """
        :param num_heads:
        :param normalise:
        """
        super().__init__()
        self._key_size = key_size
        self._value_size = value_size
        self._num_heads = num_heads
        self._key_hidden_size = key_hidden_size
        self._head_size = int(self._value_size / self._num_heads)
        self._normalise = normalise
        self._query_transform = nn.Linear(self._key_size, self._num_heads *
            self._key_hidden_size, bias=False)
        self._key_transform = nn.Linear(self._key_size, self._num_heads *
            self._key_hidden_size, bias=False)
        self._value_transform = nn.Linear(self._value_size, self._num_heads *
            self._head_size, bias=False)
        self._head_transform = nn.Linear(self._num_heads * self._head_size,
            self._value_size, bias=False)

    def forward(self, queries, keys=None, values=None):
        """
        :param queries: [batch_size, N_target, key_size]
        :param keys: [batch_size, N_context, key_size]
        :param values: [batch_size, N_context, value_size]
        :return:
        """
        if keys is None:
            keys = queries
        if values is None:
            values = queries
        self._batch_size = queries.shape[0]
        self._n_target = queries.shape[1]
        self._n_context = keys.shape[1]
        queries = self._query_transform(queries).view(self._batch_size,
            self._n_target, self._num_heads, self._key_hidden_size)
        keys = self._key_transform(keys).view(self._batch_size, self.
            _n_context, self._num_heads, self._key_hidden_size)
        values = self._value_transform(values).view(self._batch_size, self.
            _n_context, self._num_heads, self._head_size)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        attention = dot_product_attention(queries, keys, values, normalise=
            self._normalise)
        attention = attention.transpose(1, 2)
        attention = attention.reshape(self._batch_size, self._n_target, -1)
        output = self._head_transform(attention)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'key_size': 4, 'value_size': 4, 'num_heads': 4,
        'key_hidden_size': 4}]
