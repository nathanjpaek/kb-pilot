import torch
from torch import nn


class TernaryTanh(nn.Module):

    def __init__(self, beta=2.0, varying_beta=True):
        super(TernaryTanh, self).__init__()
        self.beta = beta
        self.varying_beta = varying_beta

    def forward(self, x):
        m = torch.nn.Tanh()
        if self.beta >= 1.0:
            y = m(x * self.beta * 2.0 - self.beta) * 0.5
            y += -m(-x * self.beta * 2.0 - self.beta) * 0.5
        elif self.beta == 0.0:
            y = torch.sign(x)
        elif self.beta < 0:
            y = torch.nn.HardTanh(x)
        else:
            y = torch.sign(x) * (torch.abs(x) > self.beta).float()
        return y

    def set_beta(self, beta):
        if self.varying_beta:
            self.beta = beta


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
