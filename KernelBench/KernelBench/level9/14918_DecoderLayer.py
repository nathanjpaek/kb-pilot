import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def attention(q, k, v, d_k, mask=None, dropout=None):
    """
    :param q: queries, B x N_HEADS x seq_len x d_k
    :param k: keys, same dim as q
    :param v: values, same dim as q
    :param d_k: d_model/n_heads = 128/8 = 16
    :param mask: mask for padding and future steps in the scores!
    :param dropout: dropout layer if any
    :return: attention vector of shape B x N_HEADS x seq_len x d_k
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1000000000.0)
    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        None
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output, scores


class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff=2048, dropout=0.0):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, heads, d_model, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        att_output, att_weights = attention(q, k, v, self.d_k, mask, self.
            dropout)
        att_weights = att_weights.detach()[:, -2:].sum(dim=1) / 2
        concat = att_output.transpose(1, 2).contiguous().view(bs, -1, self.
            d_model)
        output = self.out(concat)
        return output, att_weights


class DecoderLayer(nn.Module):

    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model)

    def forward(self, x, mask=None):
        x2 = self.norm_1(x)
        t, avg_scores = self.attn_1(x2, x2, x2, mask)
        x = x + self.dropout_1(t)
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x, avg_scores


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'heads': 4}]
