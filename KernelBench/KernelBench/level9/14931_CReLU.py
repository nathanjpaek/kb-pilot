import torch
import torch.nn.functional as F
import torch.nn as nn


class CReLU(nn.Module):

    def __init__(self):
        super(CReLU, self).__init__()

    def forward(self, x):
        return torch.cat((F.leaky_relu(x, 0.01, inplace=True), F.leaky_relu
            (-x, 0.01, inplace=True)), 1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
