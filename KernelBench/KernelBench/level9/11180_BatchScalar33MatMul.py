import torch
import torch.nn as nn


class BatchScalar33MatMul(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, scalar, mat):
        s = scalar.unsqueeze(2)
        s = s.expand_as(mat)
        return s * mat


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {}]
