import torch
import torch.nn as nn


class EuclideanLoss(nn.Module):

    def __init__(self):
        super(EuclideanLoss, self).__init__()

    def forward(self, pre, gt):
        N = pre.shape[0]
        diff = torch.sum((pre - gt).pow(2)) / (N * 2)
        return diff


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
