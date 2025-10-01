import torch
from torch import nn


class Entropy(nn.Module):

    def __init__(self):
        super(Entropy, self).__init__()

    def forward(self, x):
        plogp = x * torch.log(x)
        plogp[plogp != plogp] = 0
        return -torch.sum(plogp, dim=-1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
