import torch
from torch import nn


class SmoothL1loss_with_weight(nn.Module):

    def __init__(self):
        super(SmoothL1loss_with_weight, self).__init__()

    def forward(self, pred, targets, weights):
        assert pred.shape[0] == targets.shape[0] == weights.shape[0]
        loss = nn.SmoothL1Loss(reduction='none')(pred, targets)
        loss = loss.sum(dim=-1) * weights
        loss = loss.sum()
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
