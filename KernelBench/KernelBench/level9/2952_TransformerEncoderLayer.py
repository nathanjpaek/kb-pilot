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


class TransformerEncoderLayer(nn.Module):
    """
    Modified from torch.nn.TransformerEncoderLayer.
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
      normalize_before:
        whether to use layer_norm before the first block.

    Examples::
        >>> encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model: 'int', nhead: 'int', dim_feedforward: 'int'
        =2048, dropout: 'float'=0.1, activation: 'str'='relu',
        normalize_before: 'bool'=True) ->None:
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.0)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = nn.functional.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: 'torch.Tensor', src_mask:
        'Optional[torch.Tensor]'=None, src_key_padding_mask:
        'Optional[torch.Tensor]'=None) ->torch.Tensor:
        """
        Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional)

        Shape:
            src: (S, N, E).
            src_mask: (S, S).
            src_key_padding_mask: (N, S).
            S is the source sequence length, T is the target sequence length,
            N is the batch size, E is the feature number
        """
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask)[0]
        src = residual + self.dropout1(src2)
        if not self.normalize_before:
            src = self.norm1(src)
        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src2)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'nhead': 4}]
