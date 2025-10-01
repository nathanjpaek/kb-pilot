import math
import torch
import torch.nn.functional as F
import torch.fx
import torch.utils.data


def restricted_softmax(src, dim: 'int'=-1, margin: 'float'=0.0):
    src_max = torch.clamp(src.max(dim=dim, keepdim=True)[0], min=0.0)
    out = (src - src_max).exp()
    out = out / (out.sum(dim=dim, keepdim=True) + (margin - src_max).exp())
    return out


class Attention(torch.nn.Module):

    def __init__(self, dropout=0):
        super(Attention, self).__init__()
        self.dropout = dropout

    def forward(self, query, key, value):
        return self.compute_attention(query, key, value)

    def compute_attention(self, query, key, value):
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


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
