import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class BaseAttention(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class CosineAttention(BaseAttention):
    """Cosine Attention"""

    def __init__(self, dropout_rate=0.0, eps=1e-10, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.eps = eps

    def forward(self, q, k, v, attn_mask=None):
        """Forward

        Args:
            q (torch.Tensor): Query matrix, (B, T_q, D_q)
            k (torch.Tensor): Key matrix, (B, T_k, D_k)
            v (torch.Tensor): Value matrix, (B, T_v, D_v) T_v = T_k, D_v = D_k
            attn_mask (torch.BoolTensor | None): Mask tensor. True element will be masked.

        Returns:
            output (B, T_q, D_v); attention (B, T_q, T_k)

        Notes:
            Consine attention requires D_q = D_k, so I denote it as D here

        """
        q_norm = q / (q.norm(p=2, dim=-1, keepdim=True) + self.eps)
        k_norm = k / (k.norm(p=2, dim=-1, keepdim=True) + self.eps)
        attention = torch.bmm(q_norm, k_norm.permute(0, 2, 1))
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
