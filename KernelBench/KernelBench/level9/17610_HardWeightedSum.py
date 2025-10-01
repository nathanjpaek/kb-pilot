import torch
from torch import nn


class HardWeightedSum(nn.Module):

    def __init__(self, op_number=2, act=nn.ReLU, eps=0.0001):
        super(HardWeightedSum, self).__init__()
        shape = op_number, 1, 1, 1, 1
        self.weights = nn.Parameter(torch.ones(shape), requires_grad=True)
        self.act = act()
        self.eps = eps

    def forward(self, x):
        weights_num = self.act(self.weights)
        weights_denom = torch.sum(weights_num) + self.eps
        return torch.sum(weights_num * x / weights_denom, dim=0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
