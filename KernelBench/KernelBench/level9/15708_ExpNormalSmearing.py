import math
import torch
from torch import nn


class CosineCutoff(nn.Module):

    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0):
        super(CosineCutoff, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

    def forward(self, distances):
        if self.cutoff_lower > 0:
            cutoffs = 0.5 * (torch.cos(math.pi * (2 * (distances - self.
                cutoff_lower) / (self.cutoff_upper - self.cutoff_lower) + 
                1.0)) + 1.0)
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            cutoffs = cutoffs * (distances > self.cutoff_lower).float()
            return cutoffs
        else:
            cutoffs = 0.5 * (torch.cos(distances * math.pi / self.
                cutoff_upper) + 1.0)
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            return cutoffs


class ExpNormalSmearing(nn.Module):

    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=50,
        trainable=True):
        super(ExpNormalSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable
        self.cutoff_fn = CosineCutoff(0, cutoff_upper)
        self.alpha = 5.0 / (cutoff_upper - cutoff_lower)
        means, betas = self._initial_params()
        if trainable:
            self.register_parameter('means', nn.Parameter(means))
            self.register_parameter('betas', nn.Parameter(betas))
        else:
            self.register_buffer('means', means)
            self.register_buffer('betas', betas)

    def _initial_params(self):
        start_value = torch.exp(torch.scalar_tensor(-self.cutoff_upper +
            self.cutoff_lower))
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor([(2 / self.num_rbf * (1 - start_value)) ** -2] *
            self.num_rbf)
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(-self.betas * (torch.exp(
            self.alpha * (-dist + self.cutoff_lower)) - self.means) ** 2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
