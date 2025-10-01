import torch
import torch.nn as nn


class StatsNet(nn.Module):

    def __init__(self):
        super(StatsNet, self).__init__()

    def forward(self, x):
        x = x.view(x.data.shape[0], x.data.shape[1], x.data.shape[2] * x.
            data.shape[3])
        mean = torch.mean(x, 2)
        std = torch.std(x, 2)
        return torch.stack((mean, std), dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
