import torch
import torch.nn as nn


class BatchNorm2D_noparam(nn.Module):

    def __init__(self, eps=1e-08):
        super(BatchNorm2D_noparam, self).__init__()
        self.eps = eps

    def forward(self, x):
        _bs, _c, _h, _w = x.shape
        mean = torch.mean(x, (0, 2, 3), keepdim=True)
        var = torch.var(x, (0, 2, 3), keepdim=True)
        out = (x - mean) / (var + self.eps)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
