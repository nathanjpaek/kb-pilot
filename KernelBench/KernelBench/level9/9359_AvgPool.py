import torch
from torch import nn
import torch.nn.functional as F


class AvgPool(nn.Module):

    def forward(self, x):
        return F.avg_pool2d(x, x.shape[2:])


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
