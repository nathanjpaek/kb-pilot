import torch
from torch import Tensor
import torch.utils.data
import torch
from torch import nn
from typing import Optional


class VisionLanguageFusionModule(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout
            =dropout)

    def with_pos_embed(self, tensor, pos: 'Optional[Tensor]'):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, memory_key_padding_mask:
        'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None, query_pos:
        'Optional[Tensor]'=None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos
            ), key=self.with_pos_embed(memory, pos), value=memory,
            attn_mask=None, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt * tgt2
        return tgt


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'nhead': 4}]
