import torch
import torch.nn as nn
from itertools import product as product


class GlobalLayerNorm(nn.Module):

    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        var = torch.pow(y - mean, 2).mean(dim=1, keepdim=True).mean(dim=2,
            keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + 1e-08, 0.5
            ) + self.beta
        return gLN_y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channel_size': 4}]
