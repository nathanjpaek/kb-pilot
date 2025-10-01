import torch
import torch.nn as nn
import torch.utils.data


class MeanPool(nn.Module):

    def __init__(self):
        super(MeanPool, self).__init__()

    def forward(self, input):
        x = input.mean(dim=1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
