import torch
import torch.nn as nn


class BatchNorm(nn.Module):

    def __init__(self, num_channels):
        super().__init__()
        theta_mu = torch.zeros(num_channels)
        theta_sigma = torch.ones(num_channels)
        self.theta_mu = nn.Parameter(theta_mu)
        self.theta_sigma = nn.Parameter(theta_sigma)
        self.running_mean = None
        self.running_var = None
        self.eps = 1e-06

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0)
            if self.running_mean is None:
                self.running_mean = nn.Parameter(mean, requires_grad=False)
                self.running_var = nn.Parameter(var, requires_grad=False)
            else:
                self.running_mean.data = 0.1 * mean + 0.9 * self.running_mean
                self.running_var.data = 0.1 * var + 0.9 * self.running_var
        elif self.running_mean is None:
            mean = x.mean(dim=0)
            var = x.var(dim=0)
        else:
            mean = self.running_mean
            var = self.running_var
        x_norm = (x - mean) / (var + self.eps).sqrt()
        return x_norm * self.theta_sigma + self.theta_mu


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_channels': 4}]
