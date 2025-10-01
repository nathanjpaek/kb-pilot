import torch
import torch.nn as nn
import torch.utils.data
from typing import Optional
import torch.nn


class TransposeMultiheadAttention(nn.Module):
    """
    Wrapper for nn.MultiheadAttention which first transposes the input tensor
    from (batch_size, seq_len, feature_dim) to (seq_length, batch_size, feature_dim),
    then applies the attention and transposes the attention outputs back to the input
    shape.
    """

    def __init__(self, feature_dim: 'int', num_heads: 'int'=1):
        """
        Args:
            feature_dim (int): attention embedding dimension
            num_heads (int): number of attention heads
        """
        super().__init__()
        self._attention = nn.MultiheadAttention(embed_dim=feature_dim,
            num_heads=num_heads)
        self._attention_weights = None

    @property
    def attention_weights(self) ->Optional[torch.Tensor]:
        """
        Contains attention weights from last forward call.
        """
        return self._attention_weights

    def forward(self, x: 'torch.Tensor', mask: 'Optional[torch.Tensor]'=None
        ) ->torch.Tensor:
        """
        Args:
            x (torch.Tensor): tensor of shape (batch_size, seq_len, feature_dim)
            mask (torch.Tensor): bool tensor with shape (batch_size, seq_len).
                Sequence elements that are False are invalid.

        Returns:
            Tensor with shape (batch_size, seq_len, feature_dim)
        """
        assert x.dim(
            ) == 3, 'Requires x shape (batch_size x seq_len x feature_dim)'
        if mask is not None:
            mask[:, 0] = True
            mask = ~mask
        x = x.transpose(0, 1)
        attn_output, self._attention_weights = self._attention(x, x, x,
            key_padding_mask=mask)
        attn_output = attn_output.transpose(0, 1)
        return attn_output


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'feature_dim': 4}]
