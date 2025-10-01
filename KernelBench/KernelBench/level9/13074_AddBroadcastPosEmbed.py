import torch
import torch.nn as nn


def tensor_slice(x, begin, size):
    assert all([(b >= 0) for b in begin])
    size = [(l - b if s == -1 else s) for s, b, l in zip(size, begin, x.shape)]
    assert all([(s >= 0) for s in size])
    slices = [slice(b, b + s) for b, s in zip(begin, size)]
    return x[slices]


class AddBroadcastPosEmbed(nn.Module):

    def __init__(self, shape, embd_dim, dim=-1):
        super().__init__()
        assert dim in [-1, 1]
        self.shape = shape
        self.n_dim = n_dim = len(shape)
        self.embd_dim = embd_dim
        self.dim = dim
        assert embd_dim % n_dim == 0, f'{embd_dim} % {n_dim} != 0'
        self.emb = nn.ParameterDict({f'd_{i}': nn.Parameter(torch.randn(
            shape[i], embd_dim // n_dim) * 0.01 if dim == -1 else torch.
            randn(embd_dim // n_dim, shape[i]) * 0.01) for i in range(n_dim)})

    def forward(self, x, decode_step=None, decode_idx=None):
        embs = []
        for i in range(self.n_dim):
            e = self.emb[f'd_{i}']
            if self.dim == -1:
                e = e.view(1, *((1,) * i), self.shape[i], *((1,) * (self.
                    n_dim - i - 1)), -1)
                e = e.expand(1, *self.shape, -1)
            else:
                e = e.view(1, -1, *((1,) * i), self.shape[i], *((1,) * (
                    self.n_dim - i - 1)))
                e = e.expand(1, -1, *self.shape)
            embs.append(e)
        embs = torch.cat(embs, dim=self.dim)
        if decode_step is not None:
            embs = tensor_slice(embs, [0, *decode_idx, 0], [x.shape[0], *((
                1,) * self.n_dim), x.shape[-1]])
        return x + embs


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'shape': [4, 4], 'embd_dim': 4}]
