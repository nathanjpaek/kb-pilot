import math
import torch
import torch.nn as nn


class MultiheadAttention(nn.Module):

    def __init__(self, num_heads=4):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, key, query, value):
        b, d, n = key.size()
        _, _, m = query.size()
        _, do, _ = value.size()
        key = key.reshape(b * self.num_heads, d // self.num_heads, n)
        query = query.reshape(b * self.num_heads, d // self.num_heads, m)
        value = value.reshape(b * self.num_heads, do // self.num_heads, m)
        affinity = torch.bmm(key.transpose(1, 2), query)
        weight = torch.softmax(affinity / math.sqrt(d), dim=1)
        output = torch.bmm(value, weight)
        output = output.reshape(b, -1, m)
        weight = weight.reshape(b, self.num_heads, n, m)
        return output, weight


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {}]
