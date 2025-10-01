import torch
from torch import nn


class SelfAttentionUnit(nn.Module):

    def __init__(self, embed_dim, num_heads, max_len, dropout=0.8, bias=
        False, skip_connection=True):
        super(SelfAttentionUnit, self).__init__()
        self.skip_connection = skip_connection
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=
            num_heads, dropout=dropout, bias=bias)
        self.act = nn.ReLU()
        self.ln = nn.LayerNorm([max_len, embed_dim])

    def forward(self, x):
        x = x.permute(1, 0, 2)
        res, _ = self.attn(key=x, value=x, query=x)
        res = self.act(res)
        if self.skip_connection:
            res = res + x
        res = res.permute(1, 0, 2)
        return self.ln(res)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'embed_dim': 4, 'num_heads': 4, 'max_len': 4}]
