import torch
import torch.nn as nn


class ShuffleCatAlt(nn.Module):

    def forward(self, a, b):
        assert a.size() == b.size()
        n, c, h, w = a.size()
        x = torch.zeros(n, c * 2, h, w, dtype=a.dtype, device=a.device)
        x[:, ::2] = a
        x[:, 1::2] = b
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
