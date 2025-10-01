import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads, p_drop=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_hidden = d_model // n_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p_drop)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        q = self.q_linear(q).view(bs, -1, self.n_heads, self.d_hidden)
        k = self.k_linear(k).view(bs, -1, self.n_heads, self.d_hidden)
        v = self.v_linear(v).view(bs, -1, self.n_heads, self.d_hidden)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_hidden)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask is False, -1000000000.0)
        scores = F.softmax(scores, dim=-1)
        attn = torch.matmul(scores, v)
        concat = attn.transpose(1, 2).reshape(bs, -1, self.d_model)
        concat = self.dropout(concat)
        return self.fc(concat)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'n_heads': 4}]
