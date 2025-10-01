import torch
import torch.nn as nn


class RestrictionLoss(nn.Module):

    def __init__(self, otherbar=0):
        super().__init__()
        self.otherbar = otherbar

    def forward(self, predict):
        loss = torch.sum(((self.otherbar - predict) * (1 - predict)) ** 2)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
