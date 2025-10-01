import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """A simple self-attention solution."""

    def __init__(self, data_dim, dim_q):
        super(SelfAttention, self).__init__()
        self._layers = []
        self._fc_q = nn.Linear(data_dim, dim_q)
        self._layers.append(self._fc_q)
        self._fc_k = nn.Linear(data_dim, dim_q)
        self._layers.append(self._fc_k)

    def forward(self, input_data):
        _b, _t, k = input_data.size()
        queries = self._fc_q(input=input_data)
        keys = self._fc_k(input=input_data)
        dot = torch.bmm(queries, keys.transpose(1, 2))
        scaled_dot = torch.div(dot, torch.sqrt(torch.tensor(k).float()))
        return scaled_dot

    @property
    def layers(self):
        return self._layers


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'data_dim': 4, 'dim_q': 4}]
