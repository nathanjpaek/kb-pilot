import torch
import torch.nn.functional as F
from typing import *
import torch.utils.data
import torch.nn as nn
import torch.onnx.operators
import torch.optim


class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, 
            1, 1))
        nn.init.constant_(self.weight, 0.1)

    def forward(self, edges):
        edges = (edges * F.softmax(self.weight, dim=1)).sum(dim=1)
        return edges

    def extra_repr(self) ->str:
        return 'ConV {}'.format(self.weight.size())


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
