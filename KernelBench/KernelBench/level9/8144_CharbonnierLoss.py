import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):

    def __init__(self):
        super(CharbonnierLoss, self).__init__()

    def forward(self, pre, gt):
        N = pre.shape[0]
        diff = torch.sum(torch.sqrt((pre - gt).pow(2) + 0.001 ** 2)) / N
        return diff


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
