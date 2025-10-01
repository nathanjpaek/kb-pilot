import torch
import torch.nn as nn
import torch.optim


def kuma_reparametrization(a, b):
    u = torch.rand_like(a)
    k = (1 - (1 - u) ** (1 / (b + 1e-08))) ** (1 / (a + 1e-08))
    return k


class Rectifier(nn.Module):

    def __init__(self, l=-0.1, r=1.1):
        super().__init__()
        self.l = l
        self.r = r
        self.eps = 1e-07

    def forward(self, x, l=None, r=None):
        l = l if l is not None else self.l
        r = r if r is not None else self.r
        t = l + (r - l) * x
        t = torch.nn.functional.hardtanh(t, 0, 1)
        return t


class HardKumaBinarizer(nn.Module):

    def __init__(self, l=-0.1, r=1.1):
        super().__init__()
        self.rectifier = Rectifier(l, r)

    def forward(self, a, b, l=None, r=None):
        k = kuma_reparametrization(torch.exp(a), torch.exp(b))
        t = self.rectifier(k, l, r)
        return t


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
