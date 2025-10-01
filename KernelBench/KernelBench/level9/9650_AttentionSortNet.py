import torch
from torch.nn import functional as F
from functools import partial
from torch import nn


def bucket(buckets, t, dim=1):
    shape = list(t.shape)
    shape[dim:dim + 1] = [buckets, -1]
    return t.reshape(*shape)


def expand_dim(t, dim, k):
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def expand_batch_and_merge_head(b, t):
    shape = list(t.squeeze(0).shape)
    t = expand_dim(t, 0, b)
    shape[0] = shape[0] * b
    return t.reshape(*shape)


def log(t, eps=1e-06):
    return torch.log(t + eps)


def sample_gumbel(shape, device, dtype, eps=1e-06):
    u = torch.empty(shape, device=device, dtype=dtype).uniform_(0, 1)
    return -log(-log(u, eps), eps)


def sinkhorn_sorting_operator(r, n_iters=8):
    r.shape[1]
    for _ in range(n_iters):
        r = r - torch.logsumexp(r, dim=2, keepdim=True)
        r = r - torch.logsumexp(r, dim=1, keepdim=True)
    return torch.exp(r)


def gumbel_sinkhorn(r, n_iters=8, temperature=0.7):
    r = log(r)
    gumbel = sample_gumbel(r.shape, r.device, r.dtype)
    r = (r + gumbel) / temperature
    return sinkhorn_sorting_operator(r, n_iters)


class AttentionSortNet(nn.Module):

    def __init__(self, heads, buckets, dim, non_permutative, temperature,
        sinkhorn_iter, n_sortcut=0):
        super().__init__()
        self.heads = heads
        self.buckets = buckets
        self.dim = dim
        self.non_permutative = non_permutative
        self.temperature = temperature
        self.sinkhorn_iter = sinkhorn_iter
        self.n_sortcut = n_sortcut
        self.q_pos_emb = nn.Parameter(torch.randn(1, heads, buckets if 
            n_sortcut == 0 else 1, dim))
        self.k_pos_emb = nn.Parameter(torch.randn(1, heads, buckets, dim))

    def forward(self, q, k):
        bh, *_, buckets, device, dtype, _dim = (*q.shape, self.buckets, q.
            device, q.dtype, self.dim)
        b = bh // self.heads
        b_q = bucket(buckets, q) if self.n_sortcut == 0 else bucket(1, q)
        b_k = bucket(buckets, k)
        pos_q, pos_k = map(partial(expand_batch_and_merge_head, b), (self.
            q_pos_emb, self.k_pos_emb))
        sq = b_q.mean(dim=2) + pos_q
        sk = b_k.mean(dim=2) + pos_k
        R = torch.einsum('bie,bje->bij', sq, sk)
        if self.n_sortcut > 0:
            values, indices = torch.topk(R, self.n_sortcut)
            values = values.reshape(bh, self.n_sortcut, -1)
            indices = indices.reshape(bh, self.n_sortcut, -1)
            R = torch.zeros(bh, self.n_sortcut, buckets, device=device,
                dtype=dtype).scatter(2, indices, values)
        return R.softmax(dim=-1) if self.non_permutative else gumbel_sinkhorn(F
            .relu(R), self.sinkhorn_iter, self.temperature)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'heads': 4, 'buckets': 4, 'dim': 4, 'non_permutative': 4,
        'temperature': 4, 'sinkhorn_iter': 4}]
