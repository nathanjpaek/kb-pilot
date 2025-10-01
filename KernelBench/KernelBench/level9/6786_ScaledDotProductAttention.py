import torch
import numpy as np
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):

    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None, rpe_q=None, rpe_v=None):
        """
        Args:
            q: query (*, query_len, dim)
            k: key (*, key_len, dim)
            v: value (*, key_len, dim)
            mask: (*, query_len, key_len), True will be masked out
            rpe_q : (query_len, key_len, dim)
            rpe_v : (query_len, key_len, dim)
        Returns:
            context: (*, query_len, dim)
            alignment: (*, query_len, key_len)
        """
        dim = q.shape[-1]
        q /= dim ** 0.5
        energy = q @ k.transpose(-2, -1)
        if rpe_q is not None:
            energy += torch.einsum('...qd,qkd->...qk', q, rpe_q)
        if mask is not None:
            energy = energy.masked_fill(mask, np.NINF)
        alignment = torch.softmax(energy, dim=-1)
        context = self.dropout(alignment) @ v
        if rpe_v is not None:
            context += torch.einsum('...qk,qkd->...qd', alignment, rpe_v)
        return context, alignment


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dropout': 0.5}]
