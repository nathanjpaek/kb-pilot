import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        input = 1 - F.relu(1 - input)
        return input


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
