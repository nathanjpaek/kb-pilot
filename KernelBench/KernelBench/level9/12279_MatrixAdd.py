import torch
import torch.nn as nn
import torch.autograd


class MatrixAdd(nn.Module):

    def __init__(self):
        super(MatrixAdd, self).__init__()

    def forward(self, x, y):
        z = torch.add(x, y)
        return z


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
