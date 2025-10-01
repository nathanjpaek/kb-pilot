import math
import torch
from torch import Tensor
from torch.nn import Linear
import torch.nn.functional as F
from torch.nn import Parameter
import torch.utils.data


def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


def kaiming_uniform(tensor, fan, a):
    if tensor is not None:
        bound = math.sqrt(6 / ((1 + a ** 2) * fan))
        tensor.data.uniform_(-bound, bound)


def restricted_softmax(src, dim=-1, margin=0):
    src_max = torch.clamp(src.max(dim=dim, keepdim=True)[0], min=0)
    out = (src - src_max).exp()
    out = out / (out.sum(dim=dim, keepdim=True) + (margin - src_max).exp())
    return out


class Linear(torch.nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, bias=True):
        super(Linear, self).__init__()
        assert in_channels % groups == 0 and out_channels % groups == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.weight = Parameter(Tensor(groups, in_channels // groups, 
            out_channels // groups))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        kaiming_uniform(self.weight, fan=self.weight.size(1), a=math.sqrt(5))
        uniform(self.weight.size(1), self.bias)

    def forward(self, src):
        if self.groups > 1:
            size = list(src.size())[:-1]
            src = src.view(-1, self.groups, self.in_channels // self.groups)
            src = src.transpose(0, 1).contiguous()
            out = torch.matmul(src, self.weight)
            out = out.transpose(1, 0).contiguous()
            out = out.view(*(size + [self.out_channels]))
        else:
            out = torch.matmul(src, self.weight.squeeze(0))
        if self.bias is not None:
            out += self.bias
        return out

    def __repr__(self):
        return '{}({}, {}, groups={}, bias={})'.format(self.__class__.
            __name__, self.in_channels, self.out_channels, self.groups, 
            self.bias is not None)


class Attention(torch.nn.Module):

    def __init__(self, dropout=0):
        super(Attention, self).__init__()
        self.dropout = dropout

    def forward(self, query, key, value):
        assert query.dim() == key.dim() == value.dim() >= 2
        assert query.size(-1) == key.size(-1)
        assert key.size(-2) == value.size(-2)
        score = torch.matmul(query, key.transpose(-2, -1))
        score = score / math.sqrt(key.size(-1))
        score = restricted_softmax(score, dim=-1)
        score = F.dropout(score, p=self.dropout, training=self.training)
        return torch.matmul(score, value)

    def __repr__(self):
        return '{}(dropout={})'.format(self.__class__.__name__, self.dropout)


class MultiHead(Attention):

    def __init__(self, in_channels, out_channels, heads=1, groups=1,
        dropout=0, bias=True):
        super(MultiHead, self).__init__(dropout)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.groups = groups
        self.bias = bias
        assert in_channels % heads == 0 and out_channels % heads == 0
        assert in_channels % groups == 0 and out_channels % groups == 0
        assert max(groups, self.heads) % min(groups, self.heads) == 0
        self.lin_q = Linear(in_channels, out_channels, groups, bias)
        self.lin_k = Linear(in_channels, out_channels, groups, bias)
        self.lin_v = Linear(in_channels, out_channels, groups, bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_q.reset_parameters()
        self.lin_k.reset_parameters()
        self.lin_v.reset_parameters()

    def forward(self, query, key, value):
        assert query.dim() == key.dim() == value.dim() >= 2
        assert query.size(-1) == key.size(-1) == value.size(-1)
        assert key.size(-2) == value.size(-2)
        query = self.lin_q(query)
        key = self.lin_k(key)
        value = self.lin_v(value)
        size = list(query.size())[:-2]
        out_channels_per_head = self.out_channels // self.heads
        query_size = size + [query.size(-2), self.heads, out_channels_per_head]
        query = query.view(*query_size).transpose(-2, -3)
        key_size = size + [key.size(-2), self.heads, out_channels_per_head]
        key = key.view(*key_size).transpose(-2, -3)
        value_size = size + [value.size(-2), self.heads, out_channels_per_head]
        value = value.view(*value_size).transpose(-2, -3)
        out = super(MultiHead, self).forward(query, key, value)
        out = out.transpose(-3, -2).contiguous()
        out = out.view(*(size + [query.size(-2), self.out_channels]))
        return out

    def __repr__(self):
        return '{}({}, {}, heads={}, groups={}, dropout={}, bias={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.heads, self.groups, self.dropout, self.bias)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
