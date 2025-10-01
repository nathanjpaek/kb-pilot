import torch
from typing import Tuple
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi Head Attention module. https://arxiv.org/abs/1706.03762
    This version has no normalization module and suppose self-attention
    """

    def __init__(self, hidden_dim: 'int', heads: 'int', dropout_rate: 'float'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.linear_kvq = nn.Conv1d(self.hidden_dim, self.hidden_dim * 3, 1,
            bias=False)
        self.linear = nn.Conv1d(self.hidden_dim, self.hidden_dim, 1, bias=False
            )
        if 0 < dropout_rate < 1:
            self.drop_out = nn.Dropout(dropout_rate)
        else:
            self.drop_out = None
        self.layernorm = nn.GroupNorm(1, self.hidden_dim)

    def forward(self, input: 'torch.tensor', mask: 'torch.tensor'=None
        ) ->Tuple[torch.tensor, torch.tensor]:
        k, v, q = self.linear_kvq(input).chunk(3, 1)
        k, v, q = [torch.cat(x.chunk(self.heads, 1), dim=0) for x in [k, v, q]]
        if mask is not None:
            mask = mask.repeat(self.heads, 1)
        x, att = self.scale_dot_att(k, v, q, att_mask=mask)
        x = torch.cat(x.chunk(self.heads, 0), dim=1)
        x = self.linear(x)
        if self.drop_out is not None:
            x = self.drop_out(x)
        x = self.layernorm(x + input)
        return x, att

    @staticmethod
    def scale_dot_att(k: 'torch.tensor', v: 'torch.tensor', q:
        'torch.tensor', att_mask: 'torch.tensor') ->torch.tensor:
        att = torch.bmm(k.transpose(1, 2), q) / k.size(1) ** 0.5
        if att_mask is not None:
            att_mask = att_mask.unsqueeze(1)
            att.data.masked_fill_(att_mask.transpose(1, 2).data, -float('inf'))
        att = F.softmax(att, 1)
        if att_mask is not None:
            att.data.masked_fill_(att_mask.data, 0)
        return torch.bmm(v, att), att


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_dim': 4, 'heads': 4, 'dropout_rate': 0.5}]
