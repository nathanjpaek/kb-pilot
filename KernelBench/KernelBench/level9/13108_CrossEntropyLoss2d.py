import torch
from torch import nn


class CrossEntropyLoss2d(nn.Module):
    """This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class."""

    def __init__(self, weight=None, ignore_index=-100):
        super().__init__()
        self.CE = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, output, target):
        loss = self.CE(output, target)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
