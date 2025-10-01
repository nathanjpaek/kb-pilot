import torch
import torch.utils.data
import torch.nn as nn


class L2(nn.Module):

    def __init__(self):
        super(L2, self).__init__()

    def forward(self, x, target):
        return torch.mean(torch.sum((x - target) ** 2, (1, 2, 3)))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
