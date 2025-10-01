import torch
from torch import nn


class ChannelPool(nn.Module):

    def forward(self, x):
        return torch.mean(x, 1).unsqueeze(1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
