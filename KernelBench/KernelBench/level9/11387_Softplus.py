import torch
import numpy as np
from torch.utils.data import Dataset as Dataset
import torch.nn as nn
import torch.utils.data


def activation_shifting(activation):

    def shifted_activation(x):
        return activation(x) - activation(torch.zeros_like(x))
    return shifted_activation


def cauchy_softplus(x):
    pi = np.pi
    return (x * pi - torch.log(x ** 2 + 1) + 2 * x * torch.atan(x)) / (2 * pi)


def gaussian_softplus(x):
    z = np.sqrt(np.pi / 2)
    return (z * x * torch.erf(x / np.sqrt(2)) + torch.exp(-x ** 2 / 2) + z * x
        ) / (2 * z)


def gaussian_softplus2(x):
    z = np.sqrt(np.pi / 2)
    return (z * x * torch.erf(x / np.sqrt(2)) + torch.exp(-x ** 2 / 2) + z * x
        ) / z


def get_softplus(softplus_type='softplus', zero_softplus=False):
    if softplus_type == 'softplus':
        act = nn.functional.softplus
    elif softplus_type == 'gaussian_softplus':
        act = gaussian_softplus
    elif softplus_type == 'gaussian_softplus2':
        act = gaussian_softplus2
    elif softplus_type == 'laplace_softplus':
        act = gaussian_softplus
    elif softplus_type == 'cauchy_softplus':
        act = cauchy_softplus
    else:
        raise NotImplementedError(
            f'softplus type {softplus_type} not supported.')
    if zero_softplus:
        act = activation_shifting(act)
    return act


class Softplus(nn.Module):

    def __init__(self, softplus_type='softplus', zero_softplus=False):
        super(Softplus, self).__init__()
        self.softplus_type = softplus_type
        self.zero_softplus = zero_softplus

    def forward(self, x):
        return get_softplus(self.softplus_type, self.zero_softplus)(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
