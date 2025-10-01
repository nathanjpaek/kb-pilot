import torch
import torch.nn as nn
import torch.nn.functional
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


class SimpleModel(nn.Module):

    def __init__(self):
        super(SimpleModel, self).__init__()

    def forward(self, x):
        return x * 2


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
