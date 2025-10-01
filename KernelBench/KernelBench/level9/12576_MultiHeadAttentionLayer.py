import math
import torch
import torch.nn.functional as F
import torch.nn as nn


class Layer(nn.Module):

    def __init__(self, name):
        super(Layer, self).__init__()
        self.name = name


class MultiHeadAttentionLayer(Layer):

    def __init__(self, n_heads, d_src, d_tgt, dropout, name='None'):
        super(MultiHeadAttentionLayer, self).__init__(name)
        self.d_src = d_src
        self.d_tgt_ = d_tgt
        self.d_k = d_tgt // n_heads
        self.d_v = d_src // n_heads
        self.h = n_heads
        self.dropout = dropout
        self.q_linear = nn.Linear(d_tgt, d_tgt)
        self.v_linear = nn.Linear(d_src, d_src)
        self.k_linear = nn.Linear(d_src, d_tgt)
        self.drop1 = nn.Dropout(self.dropout)
        self.output_layer = nn.Linear(d_src, d_src)
        assert self.d_k * n_heads == d_tgt
        assert self.d_v * n_heads == d_src

    def forward(self, x_q, x_k, x_v, logger_conf=None, mask=None):
        bs = x_k.size(0)
        k = self.k_linear(x_k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(x_q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(x_v).view(bs, -1, self.h, self.d_v)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        weighted_values = self.attention(q, k, v, self.d_k, logger_conf,
            mask, self.drop1)
        concat = weighted_values.transpose(1, 2).contiguous().view(bs, -1,
            self.d_src)
        output = torch.transpose(self.output_layer(concat), 1, 2)
        return output

    def attention(self, q, k, v, d_k, logger_conf=None, mask=None, dropout=None
        ):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            thing = mask is True
            scores = scores.masked_fill(thing, -1000000000.0)
        scores = F.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        return torch.matmul(scores, v)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_heads': 4, 'd_src': 4, 'd_tgt': 4, 'dropout': 0.5}]
