import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):

    def __init__(self, num_q_channels: 'int', num_kv_channels: 'int',
        num_heads: 'int', dropout: 'float'):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=num_q_channels,
            num_heads=num_heads, kdim=num_kv_channels, vdim=num_kv_channels,
            dropout=dropout, batch_first=True)

    def forward(self, x_q, x_kv, pad_mask=None, attn_mask=None):
        return self.attention(x_q, x_kv, x_kv, key_padding_mask=pad_mask,
            attn_mask=attn_mask)[0]


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'num_q_channels': 4, 'num_kv_channels': 4, 'num_heads': 4,
        'dropout': 0.5}]
