import torch
import torch.utils.data
import torch.nn as nn


class L1(nn.Module):

    def __init__(self, eps=1e-06):
        super(L1, self).__init__()
        self.eps = eps

    def forward(self, x, target):
        diff = x - target
        return torch.mean(torch.sum(torch.sqrt(diff * diff + self.eps), (1,
            2, 3)))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
