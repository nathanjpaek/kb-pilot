import torch
import numpy as np
from torch import nn


class SineLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=True, is_first=False,
        omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                lim = 1 / self.in_features
            else:
                lim = np.sqrt(6 / self.in_features) / self.omega_0
            self.linear.weight.uniform_(-lim, lim)

    def forward(self, _input):
        return torch.sin(self.omega_0 * self.linear(_input))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
