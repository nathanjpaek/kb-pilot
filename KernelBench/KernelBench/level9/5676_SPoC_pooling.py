import torch
import torch.nn as nn


class SPoC_pooling(nn.Module):

    def __init__(self):
        super(SPoC_pooling, self).__init__()

    def forward(self, x):
        dim = x.size()
        pool = nn.AvgPool2d(dim[-1])
        x = pool(x)
        return x.view(dim[0], dim[1])


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
