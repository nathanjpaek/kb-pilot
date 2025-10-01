import math
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


class MaskedDirectMultiheadAttention(nn.Module):

    def __init__(self, d_in, d_out, heads, d_k=32, dropout=0.1):
        super(MaskedDirectMultiheadAttention, self).__init__()
        self.heads = heads
        self.scaling = 1 / math.sqrt(d_k)
        self.to_query = nn.Linear(d_in, heads * d_k)
        self.to_key = nn.Linear(d_in, heads * d_k)
        self.to_value = nn.Linear(d_out, d_out)
        self.to_out = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, query, key, value, mask):
        batch, N, L = value.shape[:3]
        q = self.to_query(query).view(batch, L, self.heads, -1).permute(0, 
            2, 1, 3)
        k = self.to_key(key).view(batch, L, self.heads, -1).permute(0, 2, 1, 3)
        v = self.to_value(value).view(batch, N, L, self.heads, -1).permute(
            0, 3, 1, 2, 4)
        q = q * self.scaling
        attention = torch.matmul(q, k.transpose(-2, -1))
        attention = attention.masked_fill(mask < 0.5, torch.finfo(q.dtype).min)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        out = torch.einsum('bhij,bhnjk->bhnik', attention, v)
        out = out.permute(0, 2, 3, 1, 4).contiguous().view(batch, N, L, -1)
        out = self.to_out(out)
        return out


class Str2MSA(nn.Module):

    def __init__(self, d_msa=64, d_state=32, inner_dim=32, r_ff=4, distbin=
        [8.0, 12.0, 16.0, 20.0], p_drop=0.1):
        super(Str2MSA, self).__init__()
        self.distbin = distbin
        n_att_head = len(distbin)
        self.norm_state = LayerNorm(d_state)
        self.norm1 = LayerNorm(d_msa)
        self.attn = MaskedDirectMultiheadAttention(d_state, d_msa,
            n_att_head, d_k=inner_dim, dropout=p_drop)
        self.dropout1 = nn.Dropout(p_drop, inplace=True)
        self.norm2 = LayerNorm(d_msa)
        self.ff = FeedForwardLayer(d_msa, d_msa * r_ff, p_drop=p_drop)
        self.dropout2 = nn.Dropout(p_drop, inplace=True)

    def forward(self, msa, xyz, state):
        dist = torch.cdist(xyz[:, :, 1], xyz[:, :, 1])
        mask_s = list()
        for distbin in self.distbin:
            mask_s.append(1.0 - torch.sigmoid(dist - distbin))
        mask_s = torch.stack(mask_s, dim=1)
        state = self.norm_state(state)
        msa2 = self.norm1(msa)
        msa2 = self.attn(state, state, msa2, mask_s)
        msa = msa + self.dropout1(msa2)
        msa2 = self.norm2(msa)
        msa2 = self.ff(msa2)
        msa = msa + self.dropout2(msa2)
        return msa


def get_inputs():
    return [torch.rand([4, 4, 4, 64]), torch.rand([4, 4, 4, 4]), torch.rand
        ([4, 4, 4, 32])]


def get_init_inputs():
    return [[], {}]
