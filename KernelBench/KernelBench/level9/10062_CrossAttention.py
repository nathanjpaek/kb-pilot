import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    """
    Multi head attention for Perceiver https://arxiv.org/pdf/2103.03206.pdf.
    Args:
        num_q_channels (`int`):
            Number of q channels.
        num_kv_channels (`int`):
            Number of k or v channels. k has the same channels as v.
        num_heads (`int`):
            Number of parallel attention heads.
        dropout (`nn.Module`):
            Dropout probability.
    """

    def __init__(self, num_q_channels: 'int', num_kv_channels: 'int',
        num_heads: 'int', dropout: 'float'):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=num_q_channels,
            num_heads=num_heads, kdim=num_kv_channels, vdim=num_kv_channels,
            dropout=dropout, batch_first=True)

    def forward(self, x_q, x_kv, pad_mask=None, attn_mask=None):
        """
        Forward function.
        Args:
            x_q (`Tensor`):
                Query embeddings.
            x_kv (`Tensor`):
                Key embeddings. Key equals value.
            pad_mask (`int`):
                Padding mask.
            attn_mask (`nn.Module`):
                Attention mask.
        """
        return self.attention(x_q, x_kv, x_kv, key_padding_mask=pad_mask,
            attn_mask=attn_mask)[0]


class CrossAttention(nn.Module):
    """
    Cross attention for Perceiver https://arxiv.org/pdf/2103.03206.pdf.
    Args:
        num_q_channels (`int`):
            Number of q channels.
        num_kv_channels (`int`):
            Number of k or v channels. k has the same channels as v.
        num_heads (`int`):
            Number of parallel attention heads.
        dropout (`nn.Module`):
            Dropout probability.
    """

    def __init__(self, num_q_channels: 'int', num_kv_channels: 'int',
        num_heads: 'int', dropout: 'float'):
        super().__init__()
        self.q_norm = nn.LayerNorm(num_q_channels)
        self.kv_norm = nn.LayerNorm(num_kv_channels)
        self.attention = MultiHeadAttention(num_q_channels=num_q_channels,
            num_kv_channels=num_kv_channels, num_heads=num_heads, dropout=
            dropout)

    def forward(self, x_q, x_kv, pad_mask=None, attn_mask=None):
        x_q = self.q_norm(x_q)
        x_kv = self.kv_norm(x_kv)
        return self.attention(x_q, x_kv, pad_mask=pad_mask, attn_mask=attn_mask
            )


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'num_q_channels': 4, 'num_kv_channels': 4, 'num_heads': 4,
        'dropout': 0.5}]
