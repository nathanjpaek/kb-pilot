import torch
from torch.nn import functional as F
from torch import nn


def bucket(buckets, t, dim=1):
    shape = list(t.shape)
    shape[dim:dim + 1] = [buckets, -1]
    return t.reshape(*shape)


def differentiable_topk(x, k, temperature=1.0):
    *_, n, dim = x.shape
    topk_tensors = []
    for i in range(k):
        is_last = i == k - 1
        values, indices = (x / temperature).softmax(dim=-1).topk(1, dim=-1)
        topks = torch.zeros_like(x).scatter_(-1, indices, values)
        topk_tensors.append(topks)
        if not is_last:
            x.scatter_(-1, indices, float('-inf'))
    topks = torch.cat(topk_tensors, dim=-1)
    return topks.reshape(*_, k * n, dim)


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


def cumavg(t, dim):
    r = torch.arange(1, t.shape[dim] + 1, device=t.device, dtype=t.dtype)
    expand_slice = [None] * len(t.shape)
    expand_slice[dim] = slice(None, None)
    return t.cumsum(dim=dim) / r[tuple(expand_slice)]


def mask_reordering_matrix(R, topk, temperature):
    buckets = R.shape[1]
    mask_value = max_neg_value(R)
    mask = torch.zeros(R.shape, device=R.device).bool()
    i, j = torch.triu_indices(buckets, buckets)
    mask[:, i, j + topk] = True
    R.masked_fill_(mask, mask_value)
    return differentiable_topk(R, topk, temperature)


class CausalAttentionSortNet(nn.Module):

    def __init__(self, heads, bucket_size, dim, temperature):
        super().__init__()
        self.heads = heads
        self.bucket_size = bucket_size
        self.dim = dim
        self.temperature = temperature

    def forward(self, q, k, topk=1):
        bh, *_, h, dim = *q.shape, self.heads, self.dim
        bh // h
        buckets = q.shape[1] // self.bucket_size
        kv_buckets = k.shape[1] // self.bucket_size
        q_r = bucket(buckets, cumavg(q, dim=1))
        k_r = bucket(kv_buckets, cumavg(k, dim=1))
        sq = q_r[:, :, 0]
        sk = k_r.sum(dim=2)
        sk = F.pad(sk, (0, 0, topk, 0))
        R = torch.einsum('bie,bje->bij', sq, sk) * dim ** -0.5
        return mask_reordering_matrix(R, topk, self.temperature)


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'heads': 4, 'bucket_size': 4, 'dim': 4, 'temperature': 4}]
