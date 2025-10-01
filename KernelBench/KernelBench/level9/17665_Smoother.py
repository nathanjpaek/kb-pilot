from torch.nn import Module
import torch
from torch import Tensor
from typing import Optional
import torch.nn.functional as F
from torch.nn import Dropout
from torch.nn import LayerNorm
from torch.nn import Conv1d
from torch.nn import MultiheadAttention


class Smoother(Module):
    """Convolutional Transformer Encoder Layer"""

    def __init__(self, d_model: 'int', nhead: 'int', d_hid: 'int', dropout=0.1
        ):
        super(Smoother, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.conv1 = Conv1d(d_model, d_hid, 9, padding=4)
        self.conv2 = Conv1d(d_hid, d_model, 1, padding=0)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, src: 'Tensor', src_mask: 'Optional[Tensor]'=None,
        src_key_padding_mask: 'Optional[Tensor]'=None) ->Tensor:
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = src.transpose(0, 1).transpose(1, 2)
        src2 = self.conv2(F.relu(self.conv1(src2)))
        src2 = src2.transpose(1, 2).transpose(0, 1)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'nhead': 4, 'd_hid': 4}]
