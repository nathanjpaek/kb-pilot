import torch
import torch.nn.functional as F
from torch import nn


class ShiftedSoftplus(nn.Module):
    __constants__ = ['beta', 'threshold']
    beta: 'int'
    threshold: 'int'

    def __init__(self, beta: 'int'=1, threshold: 'int'=20) ->None:
        super(ShiftedSoftplus, self).__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return F.softplus(x - 1, self.beta, self.threshold)

    def extra_repr(self) ->str:
        return 'beta={}, threshold={}'.format(self.beta, self.threshold)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
