import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


def ssp(*args, **kwargs):
    return F.softplus(*args, **kwargs) - np.log(2)


class SSP(nn.Softplus):

    def forward(self, xs):
        return ssp(xs, self.beta, self.threshold)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
