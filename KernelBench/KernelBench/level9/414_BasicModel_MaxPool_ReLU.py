import torch
import torch.nn as nn


class BasicModel_MaxPool_ReLU(nn.Module):

    def __init__(self, inplace=False):
        super().__init__()
        self.maxpool = nn.MaxPool1d(3)
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        return self.relu(self.maxpool(x)).sum(dim=1)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
