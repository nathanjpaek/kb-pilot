import torch
import torch.nn as nn
import torch.utils.data


class Stub(nn.Module):

    def __init__(self, shape):
        super(Stub, self).__init__()
        self.shape = shape
        return

    def forward(self, x):
        return x.new_ones(self.shape)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'shape': 4}]
