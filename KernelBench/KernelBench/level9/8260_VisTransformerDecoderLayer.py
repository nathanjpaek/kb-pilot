import torch
from torch import Tensor
from typing import Tuple
from typing import Optional
import torch.nn as nn


class VisTransformerDecoderLayer(nn.TransformerDecoderLayer):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
        activation='relu', layer_norm_eps=1e-05, batch_first=False, device=
        None, dtype=None) ->None:
        super(VisTransformerDecoderLayer, self).__init__(d_model, nhead,
            dim_feedforward=dim_feedforward, dropout=dropout, activation=
            activation, layer_norm_eps=layer_norm_eps, batch_first=
            batch_first, device=device, dtype=dtype)

    def forward(self, tgt: 'Tensor', memory: 'Tensor', tgt_mask:
        'Optional[Tensor]'=None, memory_mask: 'Optional[Tensor]'=None,
        tgt_key_padding_mask: 'Optional[Tensor]'=None,
        memory_key_padding_mask: 'Optional[Tensor]'=None) ->Tuple[Tensor,
        Tensor]:
        """Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, attn_output_weights = self.multihead_attn(tgt, memory, memory,
            attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn_output_weights


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'nhead': 4}]
