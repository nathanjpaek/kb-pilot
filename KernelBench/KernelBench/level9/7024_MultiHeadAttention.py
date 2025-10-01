import math
import torch
import numpy as np
import torch.nn as nn


def logistic(x, c=1, a=20, b=np.e):
    return c / (1 + a * b ** -x)


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1000000000.0)
    scores = logistic(scores)
    output = torch.matmul(scores, v)
    return output, scores


class MultiHeadAttention(nn.Module):

    def __init__(self, heads, d_model, dropout=0.1, nheads=200,
        share_params=True):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        if share_params is False:
            self.q_linear = simple_projection_3d(d_model, d_model, nheads)
            self.v_linear = simple_projection_3d(d_model, d_model, nheads)
            self.k_linear = simple_projection_3d(d_model, d_model, nheads)
        if share_params is True:
            self.q_linear = nn.Linear(d_model, d_model)
            self.v_linear = nn.Linear(d_model, d_model)
            self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        scores, w = attention(q, k, v, self.d_k, mask, self.dropout)
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output, w


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'heads': 4, 'd_model': 4}]
