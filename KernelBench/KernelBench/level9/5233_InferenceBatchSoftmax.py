import torch
import torch.nn as nn
from itertools import product as product
from math import sqrt as sqrt
from torch.nn import init as init
from torch.nn import functional as F


class InferenceBatchSoftmax(nn.Module):

    def __init__(self):
        super(InferenceBatchSoftmax, self).__init__()

    @staticmethod
    def forward(input_):
        return F.softmax(input_, dim=-1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
