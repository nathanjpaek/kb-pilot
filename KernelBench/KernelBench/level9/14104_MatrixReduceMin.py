import torch
import torch.nn as nn
import torch.autograd


class MatrixReduceMin(nn.Module):

    def __init__(self):
        super(MatrixReduceMin, self).__init__()

    def forward(self, x):
        z = torch.min(x)
        return z


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
