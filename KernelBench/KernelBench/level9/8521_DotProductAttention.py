import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class BaseAttention(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class DotProductAttention(BaseAttention):
    """Dot Product Attention"""

    def __init__(self, dropout_rate=0.0, **kwargs):
        """Initialize DotProductAttention

        Args:
            dropout_rate (float): attention dropout_rate rate
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)

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
        attention = torch.bmm(q, k.permute(0, 2, 1))
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
    return [[], {}]
