import torch
import torch.nn as nn


class ContinuousLoss_L2(nn.Module):
    """ Class to measure loss between continuous emotion dimension predictions and labels. Using l2 loss as base. """

    def __init__(self, margin=1):
        super(ContinuousLoss_L2, self).__init__()
        self.margin = margin

    def forward(self, pred, target):
        labs = torch.abs(pred - target)
        loss = labs ** 2
        loss[labs < self.margin] = 0.0
        return loss.sum()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
