import torch
import torch.nn as nn
from torch.testing._internal.common_utils import *


class MyRelu2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)


class Net2(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return MyRelu2.apply(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
