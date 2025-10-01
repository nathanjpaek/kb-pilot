import torch
from torch import nn


class Mean(nn.Module):

    def __init__(self, *args):
        super(Mean, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.mean(self.shape)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
