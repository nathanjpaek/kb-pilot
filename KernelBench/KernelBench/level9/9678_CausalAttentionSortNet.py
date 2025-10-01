import torch
from torch.nn import functional as F
from functools import partial
from torch import nn


def bucket(buckets, t, dim=1):
    shape = list(t.shape)
    shape[dim:dim + 1] = [buckets, -1]
    return t.reshape(*shape)


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


def expand_dim(t, dim, k):
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def expand_batch_and_merge_head(b, t):
    shape = list(t.squeeze(0).shape)
    t = expand_dim(t, 0, b)
    shape[0] = shape[0] * b
    return t.reshape(*shape)


def cumavg(t, dim):
    r = torch.arange(1, t.shape[dim] + 1, device=t.device)
    expand_slice = [None] * len(t.shape)
    expand_slice[dim] = slice(None, None)
    return t.cumsum(dim=dim) / r[tuple(expand_slice)]


def mask_reordering_matrix(R):
    buckets = R.shape[1]
    mask_value = max_neg_value(R)
    mask = torch.zeros(R.shape, device=R.device).bool()
    i, j = torch.triu_indices(buckets, buckets)
    mask[:, i, j + 1] = True
    R.masked_fill_(mask, mask_value)
    del mask
    R = R.softmax(dim=-1)
    R = R.tril(diagonal=-1)
    return R


class CausalAttentionSortNet(nn.Module):

    def __init__(self, heads, buckets, dim):
        super().__init__()
        self.heads = heads
        self.buckets = buckets
        self.dim = dim
        self.q_pos_emb = nn.Parameter(torch.randn(1, heads, buckets, dim))
        self.k_pos_emb = nn.Parameter(torch.randn(1, heads, buckets, dim))

    def forward(self, q, k):
        bh, *_, h, buckets, _dim = *q.shape, self.heads, self.buckets, self.dim
        b = bh // h
        pos_q, pos_k = map(partial(expand_batch_and_merge_head, b), (self.
            q_pos_emb, self.k_pos_emb))
        q_r = bucket(buckets, cumavg(q, dim=1))
        k_r = bucket(buckets, cumavg(k, dim=1))
        b_q_r = q_r[:, :, 0]
        b_k_r = k_r.sum(dim=2)
        sq = b_q_r + pos_q
        sk = b_k_r + pos_k
        sk = F.pad(sk, (0, 0, 1, 0))
        R = torch.einsum('bie,bje->bij', sq, sk)
        return mask_reordering_matrix(R)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'heads': 4, 'buckets': 4, 'dim': 4}]
