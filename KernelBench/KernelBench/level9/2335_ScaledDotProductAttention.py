import torch
from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class ScaledDotProductAttention(nn.Module):
    """
    Overview:
        Implementation of dot product attentionn with scaling.
    """

    def __init__(self, d_k: 'int', dropout: 'float'=0.0) ->None:
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: 'torch.Tensor', k: 'torch.Tensor', v:
        'torch.Tensor', mask: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
        attn = torch.matmul(q / self.d_k ** 0.5, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(~mask, -1000000000.0)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_k': 4}]
