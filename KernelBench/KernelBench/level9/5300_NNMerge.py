import torch
import torch.nn as nn


class NNMerge(nn.Module):

    def __init__(self):
        super(NNMerge, self).__init__()

    def forward(self, x):
        """ (k,D) -> (D,) """
        return torch.sum(x, -2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
