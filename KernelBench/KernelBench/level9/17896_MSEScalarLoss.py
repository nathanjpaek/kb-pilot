import torch
import torch.nn as nn
from functools import reduce


class MSEScalarLoss(nn.Module):

    def __init__(self):
        super(MSEScalarLoss, self).__init__()

    def forward(self, x, gt_map):
        return torch.pow(x.sum() - gt_map.sum(), 2) / reduce(lambda a, b: a *
            b, x.shape)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
