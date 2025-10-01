import torch
import torch.nn as nn


class HLoss(nn.Module):

    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = x * torch.log(x)
        b[torch.isnan(b)] = 0
        b = -1.0 * b.sum()
        return b


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
