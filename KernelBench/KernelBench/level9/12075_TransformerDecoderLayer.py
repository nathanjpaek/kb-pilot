import torch
import torch.nn as nn
from torch.nn import functional as F
import torch._utils


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == 'relu':
        return F.relu
    if activation == 'gelu':
        return F.gelu
    if activation == 'glu':
        return F.glu
    raise RuntimeError(f'activation should be relu/gelu, not {activation}.')


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=None, nhead=1, dropout=0.1,
        activation='relu'):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout
            =dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def forward(self, query, memory, position_embedding=None):
        if position_embedding is not None:
            query = torch.cat([query, position_embedding.flatten(2).permute
                (1, 0, 2)], dim=0)
            memory = torch.cat([memory, position_embedding.flatten(2).
                permute(1, 0, 2)], dim=0)
        tgt = self.multihead_attn(query=query, key=memory, value=memory)[0]
        tgt = memory + self.dropout1(tgt)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4}]
