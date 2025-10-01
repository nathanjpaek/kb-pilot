import torch
from torch import nn as nn


class LogisticRegression(torch.nn.Module):

    def __init__(self, **kwargs):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        xin = x.flatten()[:, None]
        output = torch.sigmoid(self.linear(xin))
        return output.reshape(x.shape)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
