import torch
import torch.nn as nn


class LearnedSigmoid(nn.Module):

    def __init__(self, slope=1):
        super().__init__()
        self.q = torch.nn.Parameter(torch.ones(slope))
        self.q.requiresGrad = True

    def forward(self, x):
        return torch.multiply(torch.sigmoid(x), self.q)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
