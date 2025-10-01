import torch
import torch.nn as nn


class MaxPool2d(nn.Module):

    def __init__(self):
        super(MaxPool2d, self).__init__()
        self.layer = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x = self.layer(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
