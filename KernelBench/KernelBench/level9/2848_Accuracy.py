import torch
from torch.nn import *
from torch.optim import *
from torch.optim.lr_scheduler import *


class Accuracy(torch.nn.Module):

    def __init__(self, reduction='mean', nlabels=5):
        super().__init__()
        self.reduction = reduction
        self.nlabels = nlabels

    def forward(self, input, target):
        if self.nlabels == 1:
            pred = input.sigmoid().gt(0.5).type_as(target)
        else:
            pred = input.argmax(1)
        acc = pred == target
        if self.reduction == 'mean':
            acc = acc.float().mean()
        elif self.reduction == 'sum':
            acc = acc.float().sum()
        return acc


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
