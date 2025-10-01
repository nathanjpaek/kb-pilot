from torch.nn import Module
import torch
from torch import Tensor
from typing import Optional
from typing import Tuple
import torch.nn.functional as F
from torch.nn import Dropout
from torch.nn import LayerNorm
from torch.nn import Conv1d
from torch.nn import MultiheadAttention


class Extractor(Module):
    """Convolutional Transformer Decoder Layer"""

    def __init__(self, d_model: 'int', nhead: 'int', d_hid: 'int', dropout=
        0.1, no_residual=False):
        super(Extractor, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.conv1 = Conv1d(d_model, d_hid, 9, padding=4)
        self.conv2 = Conv1d(d_hid, d_model, 1, padding=0)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        self.no_residual = no_residual

    def forward(self, tgt: 'Tensor', memory: 'Tensor', tgt_mask:
        'Optional[Tensor]'=None, memory_mask: 'Optional[Tensor]'=None,
        tgt_key_padding_mask: 'Optional[Tensor]'=None,
        memory_key_padding_mask: 'Optional[Tensor]'=None, memory_features:
        'Optional[Tensor]'=None) ->Tuple[Tensor, Optional[Tensor]]:
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, attn = self.cross_attn(tgt, memory if memory_features is None
             else memory_features, memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)
        if self.no_residual:
            tgt = self.dropout2(tgt2)
        else:
            tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = tgt.transpose(0, 1).transpose(1, 2)
        tgt2 = self.conv2(F.relu(self.conv1(tgt2)))
        tgt2 = tgt2.transpose(1, 2).transpose(0, 1)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'nhead': 4, 'd_hid': 4}]
