import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w_2(F.relu(self.w_1(x)))


class DecoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, padding_mask, look_ahead_mask):
        self_attention_output = self.dropout1(self.self_attention(x, x, x,
            look_ahead_mask))
        self.layernorm1(x + self_attention_output)
        cross_attention_output = self.dropout2(self.cross_attention(x,
            enc_output, enc_output, padding_mask))
        ffn_input = self.layernorm2(cross_attention_output +
            cross_attention_output)
        ffn_output = self.dropout3(self.feed_forward(ffn_input))
        output = self.layernorm3(ffn_input + ffn_output)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4,
        4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'num_heads': 4, 'd_ff': 4}]
