import torch
import torch.nn as nn
import torch.nn.functional as F


class SPoC(nn.Module):

    def __init__(self):
        super(SPoC, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, (x.size(-2), x.size(-1)))

    def __repr__(self):
        return self.__class__.__name__ + '()'


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
