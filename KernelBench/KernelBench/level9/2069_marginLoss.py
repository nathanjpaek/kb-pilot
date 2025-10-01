import torch
import torch.nn as nn


class marginLoss(nn.Module):

    def __init__(self):
        super(marginLoss, self).__init__()

    def forward(self, pos, neg, margin):
        val = pos - neg + margin
        return torch.sum(torch.max(val, torch.zeros_like(val)))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
