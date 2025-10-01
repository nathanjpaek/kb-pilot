import torch
import torch.nn as nn


class WeightConvNet(nn.Module):

    def __init__(self, in_channels, groups, n_segment):
        super(WeightConvNet, self).__init__()
        self.lastlayer = nn.Conv1d(in_channels, groups, 3, padding=1)
        self.groups = groups

    def forward(self, x):
        N, _C, T = x.shape
        x = self.lastlayer(x)
        x = x.view(N, self.groups, T)
        x = x.permute(0, 2, 1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'groups': 1, 'n_segment': 4}]
