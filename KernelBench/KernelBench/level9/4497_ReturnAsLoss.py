import torch
import torch.nn as nn


class ReturnAsLoss(nn.Module):

    def __init__(self):
        super(ReturnAsLoss, self).__init__()

    def forward(self, output, y):
        """negative logarithm return"""
        return -torch.sum(torch.log(torch.sum(output * (y + 1), dim=1)))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
