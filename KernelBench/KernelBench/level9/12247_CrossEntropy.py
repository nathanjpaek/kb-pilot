import torch
from torch.nn.functional import cross_entropy
import torch.nn as nn
import torch.optim


class CrossEntropy(nn.Module):

    def __init__(self, reduce):
        super().__init__()
        self.reduce = reduce

    def forward(self, y, target, mask=None, *args, **kwargs):
        return cross_entropy(y, target.detach(), mask, self.reduce)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'reduce': 4}]
