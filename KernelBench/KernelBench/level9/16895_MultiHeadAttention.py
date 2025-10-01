import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    """

    def __init__(self, query_dim, key_dim, num_units, num_heads):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.W_query = nn.Linear(in_features=query_dim, out_features=
            num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units,
            bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=
            num_units, bias=False)

    def forward(self, query, key):
        querys = self.W_query(query)
        keys = self.W_key(key)
        values = self.W_value(key)
        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)
        scores = torch.matmul(querys, keys.transpose(2, 3))
        scores = scores / self.key_dim ** 0.5
        scores = F.softmax(scores, dim=3)
        out = torch.matmul(scores, values)
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'query_dim': 4, 'key_dim': 4, 'num_units': 4, 'num_heads': 4}]
