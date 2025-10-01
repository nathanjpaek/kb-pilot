import torch
from torch import nn
import torch.nn.functional as F


class ReluSquared(nn.Module):

    def forward(self, input):
        return F.relu(input) ** 2


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
