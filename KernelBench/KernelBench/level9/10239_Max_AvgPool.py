import torch
import torch.nn as nn
from itertools import product as product


class Max_AvgPool(nn.Module):

    def __init__(self, kernel_size=(3, 3), stride=2, padding=1, dim=128):
        super(Max_AvgPool, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride,
            padding=padding)
        self.Avgpool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride,
            padding=padding)

    def forward(self, x):
        x = self.Maxpool(x) + self.Avgpool(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
