import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1000000000.0)
    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output


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
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
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


class Building_Block(nn.Module):

    def __init__(self, input_features, hidden_features, num_heads, dropout):
        super(Building_Block, self).__init__()
        self.input_features = input_features
        self.hidden_features = hidden_features
        self.embed_dim = 100 * self.hidden_features
        self.attn = MultiHeadAttention(num_heads, self.embed_dim, dropout)
        self.pos_ffn = FeedForward(self.embed_dim, dropout=dropout)
        self.norm_1 = Norm(self.embed_dim)
        self.norm_2 = Norm(self.embed_dim)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        x2 = self.norm_1(x)
        new_x = x + self.dropout_1(self.attn(x2, x2, x2))
        x2 = self.norm_2(new_x)
        new_x = new_x + self.dropout_2(self.pos_ffn(x2))
        return new_x


def get_inputs():
    return [torch.rand([4, 400])]


def get_init_inputs():
    return [[], {'input_features': 4, 'hidden_features': 4, 'num_heads': 4,
        'dropout': 0.5}]
