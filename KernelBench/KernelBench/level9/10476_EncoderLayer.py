import math
import torch
import torch.nn.functional as F
import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear1 = nn.Parameter(torch.randn(d_model, d_model))
        self.v_linear1 = nn.Parameter(torch.randn(d_model, d_model))
        self.k_linear1 = nn.Parameter(torch.randn(d_model, d_model))
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None, dropout=None):
        bs = q.size(0)
        if q.size(1) > 230:
            self.h = 8
            self.d_k = self.d_model // self.h
        elif q.size(1) <= 138 and q.size(1) > 230:
            self.h = 4
            self.d_k = self.d_model // self.h
        elif q.size(1) <= 138 and q.size(1) > 0:
            self.h = 2
            self.d_k = self.d_model // self.h
        k = torch.matmul(k, self.k_linear1)
        k = k.view(bs, -1, self.h, self.d_k)
        q = torch.matmul(q, self.q_linear1)
        q = q.view(bs, -1, self.h, self.d_k)
        v = torch.matmul(v, self.v_linear1)
        v = v.view(bs, -1, self.h, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        k = k.transpose(-2, -1)
        scores = torch.matmul(q, k)
        scores = scores / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1000000000.0)
        scores = F.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        scores = torch.matmul(scores, v)
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output


class Norm(nn.Module):

    def __init__(self, d_model, eps=1e-06):
        super().__init__()
        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim
            =-1, keepdim=True) + self.eps) + self.bias
        return norm


class EncoderLayer(nn.Module):

    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'heads': 4}]
