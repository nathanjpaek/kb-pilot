import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer
from typing import Optional
from torch.nn.init import xavier_uniform_


class TransformerEncoderLayer(nn.Module):

    def __init__(self, dim_model, nhead, dim_feedforward=2048, dropout=0.1,
        activation='relu', layer_norm_eps=1e-05, batch_first=False):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(dim_model, nhead, dropout=
            dropout)
        self.linear1 = nn.Linear(dim_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim_model)
        self.norm1 = nn.LayerNorm(dim_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(dim_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(dim_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu
        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, queries: 'torch.Tensor', keys: 'torch.Tensor', values:
        'torch.Tensor', src_mask: 'Optional[torch.Tensor]'=None,
        src_key_padding_mask: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
        """Pass the input through the encoder layer.

        Args:

        Shape:
        """
        queries = self.norm3(queries)
        keys = self.norm3(keys)
        values = self.norm3(values)
        src2 = self.self_attn(queries, keys, values, key_padding_mask=
            src_key_padding_mask)[0]
        src = queries + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'dim_model': 4, 'nhead': 4}]
