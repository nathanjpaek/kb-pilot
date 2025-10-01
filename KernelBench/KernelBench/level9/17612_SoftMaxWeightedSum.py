import torch
from torch import nn


class SoftMaxWeightedSum(nn.Module):

    def __init__(self, op_number=2):
        super(SoftMaxWeightedSum, self).__init__()
        shape = op_number, 1, 1, 1, 1
        self.weights = nn.Parameter(torch.ones(shape), requires_grad=True)

    def forward(self, x):
        return torch.sum(torch.softmax(self.weights, dim=0) * x, dim=0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
