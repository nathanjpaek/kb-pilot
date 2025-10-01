import math
import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, embedding_size):
        super(MLP, self).__init__()
        self.dense_h_to_4h = nn.Linear(embedding_size, embedding_size * 4)
        self.dense_4h_to_h = nn.Linear(embedding_size * 4, embedding_size)
        self.act = nn.functional.gelu

    def forward(self, x):
        h = self.act(self.dense_h_to_4h(x))
        h2 = self.dense_4h_to_h(h)
        return h2


class Attention(nn.Module):

    def __init__(self, embedding_size, num_attention_heads,
        attention_dropout, residual_dropout):
        super(Attention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.size_per_head = embedding_size // num_attention_heads
        self.embedding_size = embedding_size
        self.query_key_value = nn.Linear(embedding_size, embedding_size * 3)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.resid_drop = nn.Dropout(residual_dropout)
        self.dense = nn.Linear(embedding_size, embedding_size)

    def split_heads(self, x):
        x = x.reshape([-1, self.seq_len, self.num_attention_heads, self.
            size_per_head])
        return x.permute(0, 2, 1, 3)

    def forward(self, x, kv_cache=None):
        self.seq_len = x.shape[1]
        x = self.query_key_value(x)
        q, k, v = torch.split(x, split_size_or_sections=self.embedding_size,
            dim=2)
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        if kv_cache is not None:
            pk, pv = kv_cache[0], kv_cache[1]
            k = torch.cat([pk, k], dim=-2)
            v = torch.cat([pv, v], dim=-2)
        cached_kv = torch.stack([k, v])
        attn = torch.matmul(q, k.transpose(-1, -2))
        attn = attn / math.sqrt(self.size_per_head)
        attention_mask = torch.tril(torch.ones(self.seq_len, self.seq_len,
            dtype=torch.float32, device=x.device))
        attention_mask = attention_mask.reshape([1, 1, self.seq_len, self.
            seq_len])
        attn = attn * attention_mask - 10000.0 * (1.0 - attention_mask)
        attn = nn.Softmax(dim=-1)(attn)
        attn = self.attn_drop(attn)
        y = torch.matmul(attn, v)
        y = y.permute(0, 2, 1, 3)
        y = torch.reshape(y, [-1, self.seq_len, self.embedding_size])
        y = self.resid_drop(self.dense(y))
        return y, cached_kv


class Block(nn.Module):

    def __init__(self, embedding_size, num_attention_heads,
        attention_dropout, residual_dropout):
        super(Block, self).__init__()
        self.input_layernorm = nn.LayerNorm(embedding_size, eps=1e-05)
        self.attention = Attention(embedding_size, num_attention_heads,
            attention_dropout, residual_dropout)
        self.post_attention_layernorm = nn.LayerNorm(embedding_size, eps=1e-05)
        self.mlp = MLP(embedding_size)

    def forward(self, x, kv_cache=None):
        attn, cached_kv = self.attention(self.input_layernorm(x), kv_cache=
            kv_cache)
        x = x + attn
        z = self.post_attention_layernorm(x)
        z = self.mlp(z)
        x = x + z
        return x, cached_kv


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'embedding_size': 4, 'num_attention_heads': 4,
        'attention_dropout': 0.5, 'residual_dropout': 0.5}]
