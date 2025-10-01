import torch
import torch.nn as nn


class LearnedUtility(nn.Module):

    def __init__(self, slope=0):
        super().__init__()
        self.theta_tt = torch.nn.Parameter(slope * torch.ones(1))
        self.theta_tt.requiresGrad = True

    def forward(self, x):
        return torch.multiply(self.theta_tt, x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
