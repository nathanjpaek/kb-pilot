import math
import torch
import torch.nn as nn
from torch.autograd import *


class SelfAttention(nn.Module):
    dim_in: 'int'
    dim_k: 'int'
    dim_v: 'int'

    def __init__(self, dim_in, dim_k, dim_v):
        super(SelfAttention, self).__init__()
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / math.sqrt(dim_k)

    def forward(self, x):
        _batch, _n, dim_in = x.shape
        assert dim_in == self.dim_in
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact
        dist = torch.softmax(dist, dim=-1)
        att = torch.bmm(dist, v)
        return att


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_in': 4, 'dim_k': 4, 'dim_v': 4}]
