import torch
import torch.nn as nn


class MyBatchNorm(nn.Module):

    def __init__(self, size, epsilon=1e-05):
        super(MyBatchNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(size))
        self.beta = nn.Parameter(torch.zeros(size))
        self.epsilon = epsilon

    def forward(self, x):
        var, meu = torch.var_mean(x, axis=0)
        zed_prime = (x - meu) / torch.sqrt(var + self.epsilon)
        zed_norm = self.gamma * zed_prime + self.beta
        return zed_norm


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'size': 4}]
