import torch
from torch.autograd.gradcheck import *
import torch.nn as nn
import torch.nn


class HLoss(nn.Module):

    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x, spacing):
        volumeElement = spacing.prod()
        b = x * torch.log(x)
        b = -1.0 * b.sum() * volumeElement
        return b


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
