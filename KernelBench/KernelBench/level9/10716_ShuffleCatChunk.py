import torch
import torch.nn as nn


class ShuffleCatChunk(nn.Module):

    def forward(self, a, b):
        assert a.size() == b.size()
        _n, c, _h, _w = a.size()
        a = torch.chunk(a, chunks=c, dim=1)
        b = torch.chunk(b, chunks=c, dim=1)
        x = [None] * (c * 2)
        x[::2] = a
        x[1::2] = b
        x = torch.cat(x, dim=1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
