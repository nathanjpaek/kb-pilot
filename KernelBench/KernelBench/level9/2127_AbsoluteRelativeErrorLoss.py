import torch
from torch import nn


class AbsoluteRelativeErrorLoss(nn.Module):

    def __init__(self, epsilon=0.0001):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        error = (pred - target) / (target + self.epsilon)
        return torch.abs(error)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
