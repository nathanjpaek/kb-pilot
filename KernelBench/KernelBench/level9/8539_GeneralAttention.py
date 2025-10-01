import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class BaseAttention(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class GeneralAttention(BaseAttention):
    """General Attention"""

    def __init__(self, q_dim, k_dim, dropout_rate=0.0):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(q_dim, k_dim))
        self.dropout = nn.Dropout(dropout_rate)
        self._init_weights()

    def _init_weights(self):
        torch.nn.init.xavier_uniform_(self.weights)

    def forward(self, q, k, v, attn_mask=None):
        """Forward

        Args:
            q (torch.Tensor): Query matrix, (B, T_q, D_q)
            k (torch.Tensor): Key matrix, (B, T_k, D_k)
            v (torch.Tensor): Value matrix, (B, T_v, D_v) T_v = T_k, D_v = D_k
            attn_mask (torch.BoolTensor | None): Mask tensor. True element will be masked.

        Returns:
            output (B, T_q, D_v); attention (B, T_q, T_k)
        """
        attention = q.matmul(self.weights).bmm(k.permute(0, 2, 1))
        if attn_mask is not None:
            attention.masked_fill_(attn_mask, -np.inf)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        output = attention.bmm(v)
        return output, attention


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {'q_dim': 4, 'k_dim': 4}]
