import torch
from torch import nn


class GrayLoss(nn.Module):

    def __init__(self):
        super(GrayLoss, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, x):
        y = torch.ones_like(x) / 2.0
        return 1 / self.l1(x, y)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
