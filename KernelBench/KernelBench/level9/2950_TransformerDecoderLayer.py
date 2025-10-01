import torch
from typing import Optional
from torch import nn


def _get_activation_fn(activation: 'str'):
    if activation == 'relu':
        return nn.functional.relu
    elif activation == 'gelu':
        return nn.functional.gelu
    raise RuntimeError('activation should be relu/gelu, not {}'.format(
        activation))


class TransformerDecoderLayer(nn.Module):
    """
    Modified from torch.nn.TransformerDecoderLayer.
    Add support of normalize_before,
    i.e., use layer_norm before the first block.

    Args:
      d_model:
        the number of expected features in the input (required).
      nhead:
        the number of heads in the multiheadattention models (required).
      dim_feedforward:
        the dimension of the feedforward network model (default=2048).
      dropout:
        the dropout value (default=0.1).
      activation:
        the activation function of intermediate layer, relu or
        gelu (default=relu).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(self, d_model: 'int', nhead: 'int', dim_feedforward: 'int'
        =2048, dropout: 'float'=0.1, activation: 'str'='relu',
        normalize_before: 'bool'=True) ->None:
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.0)
        self.src_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.0)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = nn.functional.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: 'torch.Tensor', memory: 'torch.Tensor', tgt_mask:
        'Optional[torch.Tensor]'=None, memory_mask:
        'Optional[torch.Tensor]'=None, tgt_key_padding_mask:
        'Optional[torch.Tensor]'=None, memory_key_padding_mask:
        'Optional[torch.Tensor]'=None) ->torch.Tensor:
        """Pass the inputs (and mask) through the decoder layer.

        Args:
          tgt:
            the sequence to the decoder layer (required).
          memory:
            the sequence from the last layer of the encoder (required).
          tgt_mask:
            the mask for the tgt sequence (optional).
          memory_mask:
            the mask for the memory sequence (optional).
          tgt_key_padding_mask:
            the mask for the tgt keys per batch (optional).
          memory_key_padding_mask:
            the mask for the memory keys per batch (optional).

        Shape:
            tgt: (T, N, E).
            memory: (S, N, E).
            tgt_mask: (T, T).
            memory_mask: (T, S).
            tgt_key_padding_mask: (N, T).
            memory_key_padding_mask: (N, S).
            S is the source sequence length, T is the target sequence length,
            N is the batch size, E is the feature number
        """
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask)[0]
        tgt = residual + self.dropout1(tgt2)
        if not self.normalize_before:
            tgt = self.norm1(tgt)
        residual = tgt
        if self.normalize_before:
            tgt = self.norm2(tgt)
        tgt2 = self.src_attn(tgt, memory, memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)[0]
        tgt = residual + self.dropout2(tgt2)
        if not self.normalize_before:
            tgt = self.norm2(tgt)
        residual = tgt
        if self.normalize_before:
            tgt = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = residual + self.dropout3(tgt2)
        if not self.normalize_before:
            tgt = self.norm3(tgt)
        return tgt


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'nhead': 4}]
