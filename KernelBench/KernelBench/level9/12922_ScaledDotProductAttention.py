import torch
import numpy as np
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        """Init.

        Args:
          attention_dropout: A scalar, dropout rate.
        """
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """Forward pass.

        Args:
          q: Queries tensor, with shape of [B, L_q, D_q]
          k: Keys tensor, with shape of [B, L_k, D_k]
          v: Values tensor, with shape of [B, L_v, D_v]
          scale: A scalar, scale factor.
          attn_mask: A binary masking tensor, with shape of [B, L_q, L_k]

        Returns:
          Context and attention tensor.
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask:
            attention = attention.masked_fill_(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return context, attention


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {}]
