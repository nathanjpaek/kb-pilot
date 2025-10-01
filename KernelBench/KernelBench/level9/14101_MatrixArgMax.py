import torch
import torch.nn as nn
import torch.autograd


class MatrixArgMax(nn.Module):

    def __init__(self):
        super(MatrixArgMax, self).__init__()

    def forward(self, x):
        z = torch.argmax(x)
        return z


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
