import torch
from torch import nn


class RelErrorLoss(nn.Module):

    def __init__(self):
        super(RelErrorLoss, self).__init__()
        self.eps = 1e-06

    def forward(self, input, target):
        return torch.mean(torch.abs(target - input) / (target + self.eps))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
