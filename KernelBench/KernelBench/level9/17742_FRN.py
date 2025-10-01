import torch
import torch.nn as nn


class FRN(nn.Module):

    def __init__(self, num_features, eps=1e-05):
        super(FRN, self).__init__()
        self.tau = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.eps = eps

    def forward(self, x):
        x = x * torch.rsqrt(torch.mean(x ** 2, dim=[2, 3], keepdim=True) +
            self.eps)
        return torch.max(self.gamma * x + self.beta, self.tau)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_features': 4}]
