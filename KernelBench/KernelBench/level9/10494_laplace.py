import torch
import torch as tr
import torch.nn as nn


class laplace(nn.Module):

    def __init__(self, lambda_=2.0):
        super(laplace, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x: 'tr.Tensor'):
        return tr.exp(-self.lambda_ * tr.abs(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
