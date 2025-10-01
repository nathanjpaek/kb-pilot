import torch
from typing import Callable
from torch import nn
from torch.nn import functional as F
import torch.distributed.nn
from collections import OrderedDict
from typing import Optional


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: 'torch.Tensor'):
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias,
            self.eps)
        return x


class ResidualAttentionBlock(nn.Module):

    def __init__(self, d_model: 'int', n_head: 'int', act_layer: 'Callable'
        =nn.GELU):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([('c_fc', nn.Linear(d_model, 
            d_model * 4)), ('gelu', act_layer()), ('c_proj', nn.Linear(
            d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)

    def attention(self, x: 'torch.Tensor', attn_mask:
        'Optional[torch.Tensor]'=None):
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: 'torch.Tensor', attn_mask:
        'Optional[torch.Tensor]'=None):
        x = x + self.attention(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'n_head': 4}]
