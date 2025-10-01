import torch
from torch import nn


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [(x // 2) for x in k]
    return p


class SM(nn.Module):

    def __init__(self, k=3, s=1):
        super(SM, self).__init__()
        self.avg = nn.AvgPool2d(k, stride=s, padding=autopad(k))
        self.max = nn.MaxPool2d(k, stride=s, padding=autopad(k))

    def forward(self, x):
        x = self.max(self.avg(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
