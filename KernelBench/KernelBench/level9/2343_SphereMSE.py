import math
import torch
from numpy import pi
from torch import nn
from torch.nn.parameter import Parameter
import torch as th
from torch.nn import Parameter
from functools import *


class SphereMSE(nn.Module):

    def __init__(self, h, w):
        super(SphereMSE, self).__init__()
        self.h, self.w = h, w
        weight = th.zeros(1, 1, h, w)
        theta_range = th.linspace(0, pi, steps=h + 1)
        dtheta = pi / h
        dphi = 2 * pi / w
        for theta_idx in range(h):
            weight[:, :, theta_idx, :] = dphi * (math.sin(theta_range[
                theta_idx]) + math.sin(theta_range[theta_idx + 1])
                ) / 2 * dtheta
        self.weight = Parameter(weight, requires_grad=False)

    def forward(self, out, target):
        return th.sum((out - target) ** 2 * self.weight) / out.size(0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'h': 4, 'w': 4}]
