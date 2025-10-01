import torch
from torch import nn


class PairwiseCrossCorrelation(nn.Module):

    def __init__(self, lambd=1):
        super().__init__()
        self.lambd = lambd

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, inp, target=None):
        inp = inp.flatten(1)
        assert len(inp) % 2 == 0
        samples1, samples2 = inp[::2], inp[1::2]
        c = samples1.T @ samples2
        on_diag = torch.diagonal(c).add_(-1)
        on_diag = torch.pow(on_diag, 2).sum()
        off_diag = self.off_diagonal(c)
        off_diag = torch.pow(off_diag, 2).sum()
        loss = on_diag + self.lambd * off_diag
        loss = loss / len(c) ** 2
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
