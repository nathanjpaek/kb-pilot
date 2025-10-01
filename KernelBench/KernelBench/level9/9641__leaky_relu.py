import torch
from torch import nn
import torch.optim
import torch.utils.data


class _leaky_relu(nn.Module):

    def __init__(self):
        super(_leaky_relu, self).__init__()

    def forward(self, x):
        x_neg = 0.1 * x
        return torch.max(x_neg, x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
