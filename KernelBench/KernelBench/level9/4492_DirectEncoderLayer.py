import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):

    def __init__(self, d_model, eps=1e-05):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + self.eps
            )
        x = self.a_2 * (x - mean)
        x /= std
        x += self.b_2
        return x


class FeedForwardLayer(nn.Module):

    def __init__(self, d_model, d_ff, p_drop=0.1):
        super(FeedForwardLayer, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(p_drop, inplace=True)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, src):
        src = self.linear2(self.dropout(F.relu_(self.linear1(src))))
        return src


class DirectMultiheadAttention(nn.Module):

    def __init__(self, d_in, d_out, heads, dropout=0.1):
        super(DirectMultiheadAttention, self).__init__()
        self.heads = heads
        self.proj_pair = nn.Linear(d_in, heads)
        self.drop = nn.Dropout(dropout, inplace=True)
        self.proj_msa = nn.Linear(d_out, d_out)
        self.proj_out = nn.Linear(d_out, d_out)

    def forward(self, src, tgt):
        B, N, L = tgt.shape[:3]
        attn_map = F.softmax(self.proj_pair(src), dim=1).permute(0, 3, 1, 2)
        attn_map = self.drop(attn_map).unsqueeze(1)
        value = self.proj_msa(tgt).permute(0, 3, 1, 2).contiguous().view(B,
            -1, self.heads, N, L)
        tgt = torch.matmul(value, attn_map).view(B, -1, N, L).permute(0, 2,
            3, 1)
        tgt = self.proj_out(tgt)
        return tgt


class DirectEncoderLayer(nn.Module):

    def __init__(self, heads, d_in, d_out, d_ff, symmetrize=True, p_drop=0.1):
        super(DirectEncoderLayer, self).__init__()
        self.symmetrize = symmetrize
        self.attn = DirectMultiheadAttention(d_in, d_out, heads, dropout=p_drop
            )
        self.ff = FeedForwardLayer(d_out, d_ff, p_drop=p_drop)
        self.drop_1 = nn.Dropout(p_drop, inplace=True)
        self.drop_2 = nn.Dropout(p_drop, inplace=True)
        self.norm = LayerNorm(d_in)
        self.norm1 = LayerNorm(d_out)
        self.norm2 = LayerNorm(d_out)

    def forward(self, src, tgt):
        B, N, L = tgt.shape[:3]
        if self.symmetrize:
            src = 0.5 * (src + src.permute(0, 2, 1, 3))
        src = self.norm(src)
        tgt2 = self.norm1(tgt)
        tgt2 = self.attn(src, tgt2)
        tgt = tgt + self.drop_1(tgt2)
        tgt2 = self.norm2(tgt.view(B * N, L, -1)).view(B, N, L, -1)
        tgt2 = self.ff(tgt2)
        tgt = tgt + self.drop_2(tgt2)
        return tgt


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'heads': 4, 'd_in': 4, 'd_out': 4, 'd_ff': 4}]
