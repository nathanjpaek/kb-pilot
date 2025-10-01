import torch
import torch as th
import torch.nn as nn


class BCEWithLogitsLoss(nn.Module):

    def __init__(self, weight=None):
        super().__init__()
        self.loss = th.nn.BCEWithLogitsLoss(weight=weight)

    def forward(self, x, target):
        return self.loss(x, target)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
