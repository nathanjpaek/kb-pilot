import torch
import torch.utils.data
import torch
import torch.nn as nn


class MyHingeLoss(nn.Module):

    def __init__(self, margin=0.0):
        nn.Module.__init__(self)
        self.m = nn.MarginRankingLoss(margin=margin)

    def forward(self, positives, negatives):
        labels = positives.new_ones(positives.size())
        return self.m(positives, negatives, labels)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
