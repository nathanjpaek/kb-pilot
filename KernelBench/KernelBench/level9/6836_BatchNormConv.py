import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class BatchNormConv(nn.Module):

    def __init__(self, num_channels, eps=1e-08):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.gamma = Parameter(torch.Tensor(num_channels))
        self.beta = Parameter(torch.Tensor(num_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)

    def forward(self, x):
        means = x.mean(dim=(0, 2, 3))
        variances = x.var(dim=(0, 2, 3))
        x = (x.permute(0, 2, 3, 1) - means) / torch.sqrt(variances + self.eps)
        x = self.gamma * x + self.beta
        return x.permute(0, 3, 1, 2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_channels': 4}]
