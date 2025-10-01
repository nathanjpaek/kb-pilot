import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadedAttention(nn.Module):

    def __init__(self, num_head, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_head == 0
        self.d_k = d_model // num_head
        self.h = num_head
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_value = nn.Linear(d_model, d_model)
        self.linear_query = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, value, mask, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        scores = scores.masked_fill(mask == 0, -1000000000.0)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask):
        nbatches = query.size(0)
        query = self.linear_query(query).view(nbatches, -1, self.h, self.d_k
            ).transpose(1, 2)
        key = self.linear_key(key).view(nbatches, -1, self.h, self.d_k
            ).transpose(1, 2)
        value = self.linear_value(value).view(nbatches, -1, self.h, self.d_k
            ).transpose(1, 2)
        mask = mask.unsqueeze(1)
        x, _attn = self.attention(query, key, value, mask, dropout=self.dropout
            )
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k
            )
        return self.linear_out(x)


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4,
        4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'num_head': 4, 'd_model': 4}]
