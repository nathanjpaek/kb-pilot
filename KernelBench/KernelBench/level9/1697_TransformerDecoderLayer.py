import torch
import torch.nn as nn
import torch.nn.functional as F


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
        attention_dropout=0.1, activation_dropout=0.1, activation='relu',
        normalize_before=True):
        super(TransformerDecoderLayer, self).__init__()
        self.normalize_before = normalize_before
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=
            attention_dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout
            =dropout)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.activation_dropout = nn.Dropout(activation_dropout)
        self.activation = {'relu': F.relu, 'gelu': F.gelu}[activation]

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
        tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
        """
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)
        tgt = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask)[0]
        tgt = residual + self.dropout(tgt)
        if not self.normalize_before:
            tgt = self.norm1(tgt)
        residual = tgt
        if self.normalize_before:
            tgt = self.norm2(tgt)
        tgt = self.multihead_attn(tgt, memory, memory, attn_mask=
            memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = residual + self.dropout(tgt)
        if not self.normalize_before:
            tgt = self.norm2(tgt)
        residual = tgt
        if self.normalize_before:
            tgt = self.norm3(tgt)
        tgt = self.activation(self.linear1(tgt))
        tgt = self.activation_dropout(tgt)
        tgt = self.linear2(tgt)
        tgt = residual + self.dropout(tgt)
        if not self.normalize_before:
            tgt = self.norm3(tgt)
        return tgt


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'nhead': 4}]
