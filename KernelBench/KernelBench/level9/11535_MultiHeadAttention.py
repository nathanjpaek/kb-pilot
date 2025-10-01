import math
import torch
import torch.nn as nn


def scaled_dot_product_attention(query, keys, values, mask=None):
    d_k = keys.shape[-1]
    dot_score = query @ keys.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        dot_score = dot_score.masked_fill(mask == 0, -1000000000.0)
    attn_score = torch.softmax(dot_score, dim=-1)
    return attn_score @ values, attn_score


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.d_model = self.num_heads * self.depth
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

    def reshape_for_multi_heads_attention(self, t):
        batch_size = t.shape[0]
        t = t.view(batch_size, -1, self.num_heads, self.depth)
        return t.transpose(1, 2)

    def forward(self, q, k, v, mask):
        batch_size = q.shape[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        q = self.reshape_for_multi_heads_attention(q)
        k = self.reshape_for_multi_heads_attention(k)
        v = self.reshape_for_multi_heads_attention(v)
        scaled_attention, _attention_weights = scaled_dot_product_attention(q,
            k, v, mask)
        scaled_attention = scaled_attention.transpose(2, 1).contiguous().view(
            batch_size, -1, self.d_model)
        return self.wo(scaled_attention)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 16, 16])]


def get_init_inputs():
    return [[], {'d_model': 4, 'num_heads': 4}]
