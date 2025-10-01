import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return 1.78718727865 * (x * torch.sigmoid(x) - 0.20662096414)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
