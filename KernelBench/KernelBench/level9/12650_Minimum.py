import torch
import torch.nn as nn
from torch import optim as optim


class Minimum(nn.Module):

    def forward(self, x, y):
        return torch.minimum(x, y)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
