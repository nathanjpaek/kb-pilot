import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.multiprocessing


class GaussianActivation(nn.Module):

    def __init__(self, a, mu, gamma_l, gamma_r):
        super(GaussianActivation, self).__init__()
        self.a = Parameter(torch.tensor(a, dtype=torch.float32))
        self.mu = Parameter(torch.tensor(mu, dtype=torch.float32))
        self.gamma_l = Parameter(torch.tensor(gamma_l, dtype=torch.float32))
        self.gamma_r = Parameter(torch.tensor(gamma_r, dtype=torch.float32))

    def forward(self, input_features):
        self.a.data = torch.clamp(self.a.data, 1.01, 6.0)
        self.mu.data = torch.clamp(self.mu.data, 0.1, 3.0)
        self.gamma_l.data = torch.clamp(self.gamma_l.data, 0.5, 2.0)
        self.gamma_r.data = torch.clamp(self.gamma_r.data, 0.5, 2.0)
        left = input_features < self.mu
        right = input_features >= self.mu
        g_A_left = self.a * torch.exp(-self.gamma_l * (input_features -
            self.mu) ** 2)
        g_A_left.masked_fill_(right, 0.0)
        g_A_right = 1 + (self.a - 1) * torch.exp(-self.gamma_r * (
            input_features - self.mu) ** 2)
        g_A_right.masked_fill_(left, 0.0)
        g_A = g_A_left + g_A_right
        return g_A


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'a': 4, 'mu': 4, 'gamma_l': 4, 'gamma_r': 4}]
