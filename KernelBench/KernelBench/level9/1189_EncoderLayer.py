import torch
import torch.nn as nn
import torch.nn.functional as F


class SPA(nn.Module):
    """ Selective parallel attention """

    def __init__(self, n_head: 'int'=8, d_v: 'int'=64):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.sk = nn.Linear(d_v, n_head * d_v)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        bs, n_head, _lq, d_v = x.size()
        u = x.sum(dim=1)
        s = self.gap(u.transpose(1, 2)).view(bs, d_v)
        v = self.sk(s)
        v = v.view(bs, n_head, d_v)
        v = self.softmax(v)
        v = v.unsqueeze(2)
        f = x * v.expand_as(x)
        return f


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature: 'float', attn_dropout: 'float'=0.1):
        super().__init__()
        self.temperature = temperature
        self.attn_dropout = attn_dropout
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1000000000.0)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'temperature=' + str(self.temperature)
        tmpstr += ', attn_dropout=' + str(self.attn_dropout)
        tmpstr += ')'
        return tmpstr


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head: 'int'=8, d_model: 'int'=512, d_k: 'int'=64,
        d_v: 'int'=64, dropout: 'float'=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        if n_head > 1:
            self.spa = SPA(n_head=n_head, d_v=d_v)
            self.fc = nn.Linear(d_v, d_model, bias=False)
        else:
            self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-06)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)
        q, attn = self.attention(q, k, v, mask=mask)
        if n_head > 1:
            q = self.spa(q)
            q = q.sum(dim=1, keepdim=True)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)
        return q, attn


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-06)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x


class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v,
            dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=
            dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input,
            enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'd_inner': 4, 'n_head': 4, 'd_k': 4, 'd_v': 4}]
