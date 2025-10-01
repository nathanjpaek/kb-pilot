import math
import torch
import torch.nn as nn


class DotProductAttention(nn.Module):

    def __init__(self, k_dim, dropout=0.1):
        super(DotProductAttention, self).__init__()
        self.scale = 1.0 / math.sqrt(k_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        :param q: (bz, q_len, q_dim)
        :param k: (bz, k_len, k_dim)
        :param v: (bz, v_len, v_dim)
        k_len == v_len  v_dim == q_dim
        :param mask: (bz, k_len)  填充部分为0
        :return: (bz, q_len, v_dim)
        """
        att_score = torch.bmm(q, k.transpose(1, 2)).mul(self.scale)
        if mask is not None:
            att_score.masked_fill_(~mask[:, None, :], -1000000000.0)
        att_weights = self.softmax(att_score)
        if self.training:
            att_weights = self.dropout(att_weights)
        att_out = torch.bmm(att_weights, v)
        return att_out


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {'k_dim': 4}]
