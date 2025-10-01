import math
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


def qkv_attention(queries, keys, values, presence=None):
    """
    Transformer-like self-attention.

    Args:
      queries: Tensor of shape [B, N, d_k].
      keys: Tensor of shape [B, M, d_k].
      values: : Tensor of shape [B, M, d_v].
      presence: None or tensor of shape [B, M].

    Returns:
      Tensor of shape [B, N, d_v]
    """
    d_k = queries.shape[-1]
    routing = torch.matmul(queries, keys.transpose(1, 2))
    if presence is not None:
        routing -= (1.0 - presence.unsqueeze(-2)) * 1e+32
    routing = F.softmax(routing / np.sqrt(d_k), -1)
    return torch.matmul(routing, values)


class MultiHeadQKVAttention(nn.Module):
    """Multi-head version of Transformer-like attention."""

    def __init__(self, d_k, d_v, n_heads):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        d_k_p = int(math.ceil(d_k / n_heads)) * n_heads
        d_v_p = int(math.ceil(d_v / n_heads)) * n_heads
        self.q_projector = nn.Linear(d_k, d_k_p)
        self.k_projector = nn.Linear(d_k, d_k_p)
        self.v_projector = nn.Linear(d_v, d_v_p)
        self.o_projector = nn.Linear(d_v_p, d_v)

    def forward(self, queries, keys, values, presence=None):
        """
        Multi-head transformer-like self-attention.

        Args:
          queries: Tensor of shape [B, N, d_k].
          keys: Tensor of shape [B, M, d_k].
          values: : Tensor of shape [B, M, d_v].
          presence: None or tensor of shape [B, M].

        Returns:
          Tensor of shape [B, N, d_v]
        """
        assert queries.shape[2] == keys.shape[2]
        assert keys.shape[1] == values.shape[1]
        if presence is not None:
            assert values.shape[:2] == presence.shape
        B, N, _d_k = queries.shape
        M, _d_v = values.shape[1:]
        H = self.n_heads
        q_p = self.q_projector(queries)
        k_p = self.k_projector(keys)
        v_p = self.v_projector(values)
        del queries, keys, values
        q = q_p.view(B, N, H, -1).permute(2, 0, 1, 3).contiguous().view(H *
            B, N, -1)
        k = k_p.view(B, M, H, -1).permute(2, 0, 1, 3).contiguous().view(H *
            B, M, -1)
        v = v_p.view(B, M, H, -1).permute(2, 0, 1, 3).contiguous().view(H *
            B, M, -1)
        if presence is not None:
            presence = presence.repeat(self.n_heads, 1)
        o = qkv_attention(q, k, v, presence)
        o = o.view(H, B, N, -1).permute(1, 2, 0, 3).contiguous().view(B, N, -1)
        return self.o_projector(o)


class MAB(nn.Module):

    def __init__(self, d, n_heads, layer_norm=False):
        super().__init__()
        self.layer_norm = layer_norm
        self.mqkv = MultiHeadQKVAttention(d_k=d, d_v=d, n_heads=n_heads)
        if layer_norm:
            self.ln0 = nn.LayerNorm(d)
            self.ln1 = nn.LayerNorm(d)
        self.fc = nn.Linear(d, d)

    def forward(self, queries, keys, presence=None):
        h = self.mqkv(queries, keys, keys, presence)
        h = h + queries
        if presence is not None:
            assert presence.shape[1] == queries.shape[1] == keys.shape[1]
            h = h * presence.unsqueeze(-1)
        if self.layer_norm:
            h = self.ln0(h)
        h = h + F.relu(self.fc(h))
        if self.layer_norm:
            h = self.ln1(h)
        return h


class PMA(nn.Module):

    def __init__(self, d, n_heads, n_seeds, layer_norm=False):
        super().__init__()
        self.mab = MAB(d=d, n_heads=n_heads, layer_norm=layer_norm)
        self.S = nn.Parameter(torch.zeros(1, n_seeds, d), requires_grad=True)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.S)

    def forward(self, x, presence=None):
        batch_size = x.shape[0]
        return self.mab(self.S.repeat(batch_size, 1, 1), x, presence)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'d': 4, 'n_heads': 4, 'n_seeds': 4}]
