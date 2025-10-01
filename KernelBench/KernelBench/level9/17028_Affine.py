import math
import torch
from torch import nn
import torch.autograd
from torch.nn import init


class Affine(nn.Module):
    """
    This module implements the affine parameters gamma and beta seen in
    Eq. 10 in Pezeshki et al. (2016). It differs from the way affine
    is used in batchnorm out of the box of PyTorch.

    Pytorch affine      : y = bn(x)*gamma + beta
    Rasmus et al. (2015): y = gamma * (bn(x) + beta)
    """

    def __init__(self, n_channels, map_size):
        super(Affine, self).__init__()
        self.map_size = map_size
        self.n_channels = n_channels
        self.gamma = nn.Parameter(torch.Tensor(self.n_channels, self.
            map_size, self.map_size))
        self.beta = nn.Parameter(torch.Tensor(self.n_channels, self.
            map_size, self.map_size))

    def forward(self, x):
        out = self.gamma * (x + self.beta)
        return out

    def reset_parameters(self) ->None:
        init.kaiming_uniform_(self.gamma, a=math.sqrt(5))
        init.kaiming_uniform_(self.beta, a=math.sqrt(5))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_channels': 4, 'map_size': 4}]
