import torch
import numpy as np
import torch.nn as nn


class RBF_activation(torch.nn.Module):

    def __init__(self, input_features):
        super(RBF_activation, self).__init__()
        self.input_features = input_features
        self.centers = nn.Parameter(torch.ones(input_features))
        self.log_sigma2 = nn.Parameter(torch.ones(input_features) * -0.2)
        self.pi = np.pi

    def forward(self, x):
        mus = self.centers.expand_as(x)
        s2 = torch.exp(self.log_sigma2.expand_as(x))
        diff = x - mus
        x = 1 / torch.sqrt(2 * self.pi * s2) * torch.exp(-torch.mul(diff,
            diff) / (2 * s2))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_features': 4}]
