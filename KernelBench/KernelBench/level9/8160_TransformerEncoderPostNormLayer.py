import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional
from torch.nn import LayerNorm


def _get_activation_fn(activation):
    if activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return F.gelu
    raise RuntimeError('activation should be relu/gelu, not {}'.format(
        activation))


class TransformerEncoderPostNormLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.0,
        activation='relu'):
        super().__init__()
        assert dropout == 0.0
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, src, src_mask: 'Optional[torch.Tensor]'=None,
        src_key_padding_mask: 'Optional[torch.Tensor]'=None):
        norm_src = self.norm1(src)
        src2 = self.self_attn(norm_src, norm_src, norm_src, attn_mask=
            src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + src2
        norm_src = self.norm2(src)
        src2 = self.linear2(self.activation(self.linear1(norm_src)))
        src = src + src2
        return src


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'nhead': 4}]
