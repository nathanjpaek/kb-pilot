import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.parallel


class PairwiseLoss(nn.Module):

    def __init__(self):
        super(PairwiseLoss, self).__init__()

    def forward(self, x, y):
        diff = x - y
        return torch.sum(diff * diff)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
