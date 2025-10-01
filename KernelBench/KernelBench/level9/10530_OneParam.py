import torch
from torch import nn
import torch.utils.data


class OneParam(nn.Module):

    def __init__(self, xdim, ydim):
        """This module computes the dynamics at a point x. That is it return the Jacobian matrix
        where each element is dy_i/dx_j
        Output is a matrix of size ydim x xdim
        """
        super(OneParam, self).__init__()
        self.W = nn.Parameter(torch.zeros(xdim, requires_grad=True))

    def forward(self, x):
        out = torch.tanh(self.W * x)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'xdim': 4, 'ydim': 4}]
