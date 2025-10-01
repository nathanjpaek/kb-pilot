import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.cuda.amp import autocast


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    @autocast()
    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -2 ** 15)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class LowRankMultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs_u = nn.Linear(d_model, int(n_head * d_k / 4), bias=False)
        self.w_qs_v = nn.Linear(int(n_head * d_k / 4), n_head * d_k, bias=False
            )
        self.w_ks_u = nn.Linear(d_model, int(n_head * d_k / 4), bias=False)
        self.w_ks_v = nn.Linear(int(n_head * d_k / 4), n_head * d_k, bias=False
            )
        self.w_vs_u = nn.Linear(d_model, int(n_head * d_k / 4), bias=False)
        self.w_vs_v = nn.Linear(int(n_head * d_k / 4), n_head * d_k, bias=False
            )
        self.fc_u = nn.Linear(n_head * d_v, int(d_model / 4), bias=False)
        self.fc_v = nn.Linear(int(d_model / 4), d_model, bias=False)
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-06)

    @autocast()
    def forward(self, q, k, v, mask=None):
        d_k, _d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, _len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        q = self.w_qs_v(self.w_qs_u(q)).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks_v(self.w_ks_u(k)).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs_v(self.w_vs_u(v)).view(sz_b, len_k, n_head, d_k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)
        q, attn = self.attention(q, k, v, mask=mask)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc_v(self.fc_u(q)))
        q += residual
        q = self.layer_norm(q)
        return q, attn


class LowRankPositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1_u = nn.Linear(d_in, int(d_in / 4), bias=False)
        self.w_1_v = nn.Linear(int(d_in / 4), d_hid)
        self.w_2_u = nn.Linear(d_hid, int(d_in / 4), bias=False)
        self.w_2_v = nn.Linear(int(d_in / 4), d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-06)
        self.dropout = nn.Dropout(dropout)

    @autocast()
    def forward(self, x):
        residual = x
        x = self.w_2_v(self.w_2_u(F.relu(self.w_1_v(self.w_1_u(x)))))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x


class LowRankEncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(LowRankEncoderLayer, self).__init__()
        self.slf_attn = LowRankMultiHeadAttention(n_head, d_model, d_k, d_v,
            dropout=dropout)
        self.pos_ffn = LowRankPositionwiseFeedForward(d_model, d_inner,
            dropout=dropout)

    @autocast()
    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input,
            enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'd_inner': 4, 'n_head': 4, 'd_k': 4, 'd_v': 4}]
