import torch
import torch.utils.data
import torch
import torch.nn as nn


class CoefficientRegularization(nn.Module):

    def __init__(self):
        super(CoefficientRegularization, self).__init__()

    def forward(self, input):
        return torch.sum(input ** 2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
