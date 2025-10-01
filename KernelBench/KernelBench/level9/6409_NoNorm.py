import torch
import torch.nn as nn


class NoNorm(nn.Module):

    def __init__(self, feat_size):
        super(NoNorm, self).__init__()
        self.bias = nn.Parameter(torch.zeros(feat_size))
        self.weight = nn.Parameter(torch.ones(feat_size))

    def forward(self, input_tensor):
        return input_tensor * self.weight + self.bias


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'feat_size': 4}]
