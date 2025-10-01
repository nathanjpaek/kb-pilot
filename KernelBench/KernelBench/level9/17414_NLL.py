import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel


class NLL(nn.Module):

    def __init__(self):
        super(NLL, self).__init__()

    def forward(self, x):
        return torch.mean(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
