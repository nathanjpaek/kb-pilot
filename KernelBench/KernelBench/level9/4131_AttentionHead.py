import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.functional import Tensor


def scaled_dot_product_attention(query: 'torch.Tensor', key: 'torch.Tensor',
    value: 'torch.Tensor') ->Tensor:
    temp = query.bmm(key.transpose(1, 2))
    scale = query.size(-1) ** 0.5
    softmax = F.softmax(temp / scale, dim=-1)
    return softmax.bmm(value)


class AttentionHead(nn.Module):

    def __init__(self, dim_in: 'int', dim_k: 'int', dim_v: 'int'):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_k)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_v)

    def forward(self, query: 'torch.Tensor', key: 'torch.Tensor', value:
        'Tensor') ->Tensor:
        return scaled_dot_product_attention(self.q(query), self.k(key),
            self.v(value))


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {'dim_in': 4, 'dim_k': 4, 'dim_v': 4}]
