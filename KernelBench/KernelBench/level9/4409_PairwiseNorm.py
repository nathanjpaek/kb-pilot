import torch
from torch import nn


class PairwiseNorm(nn.Module):

    def __init__(self, order=1, size_average=True):
        super().__init__()
        self.order = order
        self.average = size_average

    def forward(self, inp, target=None):
        inp = inp.flatten(1)
        assert len(inp) % 2 == 0
        samples1, samples2 = inp[::2], inp[1::2]
        norm = (samples1 - samples2).norm(p=self.order, dim=1).sum()
        if self.average:
            norm = norm / len(inp)
        return norm


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
