import torch
import torch.nn as nn
from torch.nn import functional as F


class StochasticClassifier(nn.Module):

    def __init__(self, num_features, num_classes, temp=0.05):
        super().__init__()
        self.mu = nn.Parameter(0.01 * torch.randn(num_classes, num_features))
        self.sigma = nn.Parameter(torch.zeros(num_classes, num_features))
        self.temp = temp

    def forward(self, x, stochastic=True):
        mu = self.mu
        sigma = self.sigma
        if stochastic:
            sigma = F.softplus(sigma - 4)
            weight = sigma * torch.randn_like(mu) + mu
        else:
            weight = mu
        weight = F.normalize(weight, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)
        score = F.linear(x, weight)
        score = score / self.temp
        return score


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_features': 4, 'num_classes': 4}]
