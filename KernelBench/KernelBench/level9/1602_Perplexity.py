import torch
from torch import nn as nn
from torch.nn.modules.loss import CrossEntropyLoss


class Perplexity(CrossEntropyLoss):
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, weight=None, size_average=None, ignore_index=-100,
        reduce=None):
        super(Perplexity, self).__init__(weight, size_average, ignore_index,
            reduce, 'mean')

    def forward(self, input, target):
        loss = super(Perplexity, self).forward(input, target)
        return torch.exp(loss)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
