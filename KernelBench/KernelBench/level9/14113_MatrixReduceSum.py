import torch
import torch.nn as nn
import torch.autograd


class MatrixReduceSum(nn.Module):

    def __init__(self):
        super(MatrixReduceSum, self).__init__()

    def forward(self, x):
        z = torch.sum(x)
        return z


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
