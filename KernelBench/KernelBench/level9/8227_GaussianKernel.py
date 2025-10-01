import math
import torch
import torch.nn as nn
import torch.utils.data


class GaussianKernel(nn.Module):

    def __init__(self, delta_var, pmaps_threshold):
        super().__init__()
        self.delta_var = delta_var
        self.two_sigma = delta_var * delta_var / -math.log(pmaps_threshold)

    def forward(self, dist_map):
        return torch.exp(-dist_map * dist_map / self.two_sigma)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'delta_var': 4, 'pmaps_threshold': 4}]
