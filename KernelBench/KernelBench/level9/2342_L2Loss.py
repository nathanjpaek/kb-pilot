import torch
from torch import nn
import torch as th
from functools import *


class L2Loss(nn.Module):

    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, grad_fake, grad_real):
        num_pixels = reduce(lambda x, y: x * y, grad_real.size())
        return th.sum(th.pow(grad_real - grad_fake, 2)) / num_pixels


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
