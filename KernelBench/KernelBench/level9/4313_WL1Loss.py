import torch
import torch.nn as nn


class WL1Loss(nn.Module):

    def __init__(self):
        super(WL1Loss, self).__init__()

    def forward(self, pred, target, weight):
        return torch.mean(weight * torch.abs(pred - target))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
