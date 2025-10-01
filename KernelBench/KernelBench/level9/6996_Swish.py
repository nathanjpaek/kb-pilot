import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x * F.sigmoid(self.beta * x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
