import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


class LayerShift(nn.Module):

    def __init__(self, init=1.0):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x - self.bias


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
