import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module."""

    def __init__(self, n_head=8, d_model=512, d_k=64, d_v=64, dropout=0.1,
        qkv_bias=False, mask_value=0):
        super().__init__()
        self.mask_value = mask_value
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.scale = d_k ** -0.5
        self.dim_k = n_head * d_k
        self.dim_v = n_head * d_v
        self.linear_q = nn.Linear(self.dim_k, self.dim_k, bias=qkv_bias)
        self.linear_k = nn.Linear(self.dim_k, self.dim_k, bias=qkv_bias)
        self.linear_v = nn.Linear(self.dim_v, self.dim_v, bias=qkv_bias)
        self.fc = nn.Linear(self.dim_v, d_model, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size, len_q, _ = q.size()
        _, len_k, _ = k.size()
        q = self.linear_q(q).view(batch_size, len_q, self.n_head, self.d_k)
        k = self.linear_k(k).view(batch_size, len_k, self.n_head, self.d_k)
        v = self.linear_v(v).view(batch_size, len_k, self.n_head, self.d_v)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 1, 3)
        logits = torch.matmul(q, k) * self.scale
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            elif mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            logits = logits.masked_fill(mask == self.mask_value, float('-inf'))
        weights = logits.softmax(dim=-1)
        weights = self.attn_drop(weights)
        attn_out = torch.matmul(weights, v).transpose(1, 2)
        attn_out = attn_out.reshape(batch_size, len_q, self.dim_v)
        attn_out = self.fc(attn_out)
        attn_out = self.proj_drop(attn_out)
        return attn_out


class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module."""

    def __init__(self, d_in, d_hid, dropout=0.1, act_layer=nn.GELU):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.act = act_layer()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """"""

    def __init__(self, d_model=512, d_inner=256, n_head=8, d_k=64, d_v=64,
        dropout=0.1, qkv_bias=False, mask_value=0, act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(n_head, d_model, d_k, d_v, qkv_bias=
            qkv_bias, dropout=dropout, mask_value=mask_value)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = PositionwiseFeedForward(d_model, d_inner, dropout=
            dropout, act_layer=act_layer)

    def forward(self, x, mask=None):
        residual = x
        x = self.norm1(x)
        x = residual + self.attn(x, x, x, mask)
        residual = x
        x = self.norm2(x)
        x = residual + self.mlp(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 512])]


def get_init_inputs():
    return [[], {}]
