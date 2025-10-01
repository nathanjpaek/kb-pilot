import torch
from torch import nn
from typing import Optional


class ScaledDotProductAttention(nn.Module):

    def __init__(self, dropout: 'Optional[float]'=None, scale: 'bool'=True):
        super(ScaledDotProductAttention, self).__init__()
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = dropout
        self.softmax = nn.Softmax(dim=2)
        self.scale = scale

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.permute(0, 2, 1))
        if self.scale:
            dimension = torch.sqrt(torch.tensor(k.shape[-1]))
            attn = attn / dimension
        if mask is not None:
            attn = attn.masked_fill(mask, -1000000000.0)
        attn = self.softmax(attn)
        if self.dropout is not None:
            attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {}]
