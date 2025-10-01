import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):

    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = d_k ** -0.5

    def forward(self, q, k, v, mask):
        x = torch.matmul(q, k.transpose(-2, -1))
        x = x if mask is None else x.masked_fill(mask, float('-inf'))
        x = torch.matmul(torch.softmax(self.scale * x, dim=-1), v)
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, d_k, d_v, h):
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.d_k = d_k
        self.d_v = d_v
        self.w_q = nn.Linear(d_model, h * d_k, bias=False)
        self.w_k = nn.Linear(d_model, h * d_k, bias=False)
        self.w_v = nn.Linear(d_model, h * d_v, bias=False)
        self.w_o = nn.Linear(h * d_v, d_model, bias=False)
        self.attention = ScaledDotProductAttention(d_k)

    def _split_into_heads(self, *xs):
        return [x.view(x.size(0), x.size(1), self.h, -1).transpose(1, 2) for
            x in xs]

    def forward(self, q, k, v, mask=None):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self._split_into_heads(q, k, v)
        x = self.attention(q, k, v, mask)
        x = x.transpose(1, 2).reshape(x.size(0), x.size(2), -1)
        x = self.w_o(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {'d_model': 4, 'd_k': 4, 'd_v': 4, 'h': 4}]
