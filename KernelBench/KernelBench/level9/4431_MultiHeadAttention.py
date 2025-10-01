import torch
from torch import Tensor
from typing import Any
import torch.nn.functional as F
from torch import nn


def ifnone(a: 'Any', b: 'Any') ->Any:
    """`a` if `a` is not None, otherwise `b`."""
    return b if a is None else a


class MultiHeadAttention(nn.Module):
    """MutiHeadAttention."""

    def __init__(self, n_heads: 'int', d_model: 'int', d_head: 'int'=None,
        resid_p: 'float'=0.0, attn_p: 'float'=0.0, bias: 'bool'=True, scale:
        'bool'=True):
        super().__init__()
        d_head = ifnone(d_head, d_model // n_heads)
        self.n_heads, self.d_head, self.scale = n_heads, d_head, scale
        self.attention = nn.Linear(d_model, 3 * n_heads * d_head, bias=bias)
        self.out = nn.Linear(n_heads * d_head, d_model, bias=bias)
        self.drop_att, self.drop_res = nn.Dropout(attn_p), nn.Dropout(resid_p)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: 'Tensor', mask: 'Tensor'=None, **kwargs):
        return self.ln(x + self.drop_res(self.out(self._apply_attention(x,
            mask=mask, **kwargs))))

    def _apply_attention(self, x: 'Tensor', mask: 'Tensor'=None):
        bs, x_len = x.size(0), x.size(1)
        wq, wk, wv = torch.chunk(self.attention(x), 3, dim=-1)
        wq, wk, wv = map(lambda x: x.view(bs, x_len, self.n_heads, self.
            d_head), (wq, wk, wv))
        wq, wk, wv = wq.permute(0, 2, 1, 3), wk.permute(0, 2, 3, 1
            ), wv.permute(0, 2, 1, 3)
        attn_score = torch.matmul(wq, wk)
        if self.scale:
            attn_score.div_(self.d_head ** 0.5)
        if mask is not None:
            attn_score = attn_score.float().masked_fill(mask, -float('inf')
                ).type_as(attn_score)
        attn_prob = self.drop_att(F.softmax(attn_score, dim=-1))
        attn_vec = torch.matmul(attn_prob, wv)
        return attn_vec.permute(0, 2, 1, 3).contiguous().contiguous().view(bs,
            x_len, -1)

    def _attention_einsum(self, x, mask=None):
        bs, x_len = x.size(0), x.size(1)
        wq, wk, wv = torch.chunk(self.attention(x), 3, dim=-1)
        wq, wk, wv = map(lambda x: x.view(bs, x_len, self.n_heads, self.
            d_head), (wq, wk, wv))
        attn_score = torch.einsum('bind,bjnd->bijn', (wq, wk))
        if self.scale:
            attn_score.mul_(1 / self.d_head ** 0.5)
        if mask is not None:
            attn_score = attn_score.float().masked_fill(mask, -float('inf')
                ).type_as(attn_score)
        attn_prob = self.drop_att(F.softmax(attn_score, dim=2))
        attn_vec = torch.einsum('bijn,bjnd->bind', (attn_prob, wv))
        return attn_vec.contiguous().view(bs, x_len, -1)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'n_heads': 4, 'd_model': 4}]
