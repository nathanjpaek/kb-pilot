import math
import torch
from torch import nn
from torch.nn import functional as F
from numpy import inf
from math import inf


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, d_qk, d_v, num_head):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.d_qk = d_qk
        self.d_v = d_v
        self.W_Q = Linear(d_model, num_head * d_qk)
        self.W_K = Linear(d_model, num_head * d_qk)
        self.W_V = Linear(d_model, num_head * d_v)
        self.W_out = Linear(d_v * num_head, d_model)

    def ScaledDotProductAttention(self, query, keys, values, mask=None):
        score = torch.matmul(query, keys.transpose(-1, -2)) / math.sqrt(self
            .d_model)
        if mask is not None:
            score.masked_fill_(mask.unsqueeze(1), -inf)
        weight = F.softmax(score, dim=-1)
        return torch.matmul(weight, values)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        query = self.W_Q(Q).view(batch_size, Q.size(1), self.num_head, self
            .d_qk)
        keys = self.W_K(K).view(batch_size, K.size(1), self.num_head, self.d_qk
            )
        values = self.W_V(V).view(batch_size, V.size(1), self.num_head,
            self.d_v)
        query.transpose_(1, 2)
        keys.transpose_(1, 2)
        values.transpose_(1, 2)
        outputs = self.ScaledDotProductAttention(query, keys, values, mask)
        del query, keys, values
        outputs = outputs.transpose(1, 2).contiguous().view(batch_size, -1,
            self.d_v * self.num_head)
        return self.W_out(outputs)

    def cal_one_vector(self, vector, memory, memory_new, i):
        batch_size = vector.size(0)
        query = self.W_Q(vector).view(batch_size, vector.size(1), self.
            num_head, self.d_qk)
        key = self.W_K(vector).view(batch_size, vector.size(1), self.
            num_head, self.d_qk)
        value = self.W_V(vector).view(batch_size, vector.size(1), self.
            num_head, self.d_v)
        query.transpose_(1, 2)
        key.transpose_(1, 2)
        value.transpose_(1, 2)
        outputs = torch.cat((key.unsqueeze(-1), value.unsqueeze(-1)), dim=-1)
        del key, value
        if memory is not None:
            if memory_new is None:
                memory_new = torch.cat((memory[:, i, ...], outputs), dim=2
                    ).unsqueeze(1)
            else:
                _m = torch.cat((memory[:, i, ...], outputs), dim=2)
                memory_new = torch.cat((memory_new, _m.unsqueeze(1)), dim=1)
        elif memory_new is None:
            memory_new = outputs.unsqueeze(1)
        else:
            memory_new = torch.cat((memory_new, outputs.unsqueeze(1)), dim=1)
        outputs = self.ScaledDotProductAttention(query, memory_new[:, i,
            ..., 0], memory_new[:, i, ..., 1])
        outputs = outputs.transpose(1, 2).contiguous().view(batch_size, -1,
            self.d_v * self.num_head)
        return self.W_out(outputs), memory_new


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {'d_model': 4, 'd_qk': 4, 'd_v': 4, 'num_head': 4}]
