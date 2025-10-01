import torch
from torch import nn
from typing import Optional


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention

    Parameters
    ----------
    scale : float
        Scale factor (sqrt(d_k))

    dropout : float
        Dropout
    """

    def __init__(self, scale: 'float', dropout: 'float'=0.5) ->None:
        super(ScaledDotProductAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q: 'torch.Tensor', K: 'torch.Tensor', V:
        'torch.Tensor', mask: 'Optional[torch.Tensor]'=None):
        """
        Parameters
        ----------
        Q : torch.Tensor (batch_size, n_heads, word_pad_len, d_k)
            Query

        K : torch.Tensor
            Key

        V : torch.Tensor
            Value

        mask : torch.Tensor (batch_size, 1, 1, word_pad_len)
            Padding mask metrix, None if it is not needed

        Returns
        -------
        context : torch.Tensor (batch_size, n_heads, word_pad_len, d_k)
            Context vector

        att : torch.Tensor (batch_size, n_heads, word_pad_len, word_pad_len)
            Attention weights
        """
        att = torch.matmul(Q / self.scale, K.transpose(2, 3))
        if mask is not None:
            att = att.masked_fill(mask == 0, -1000000000.0)
        att = self.dropout(self.softmax(att))
        context = torch.matmul(att, V)
        return context, att


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'scale': 1.0}]
