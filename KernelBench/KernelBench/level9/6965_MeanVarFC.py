import torch
import torch.nn as nn


class MeanVarFC(nn.Module):

    def __init__(self, input_shape):
        super(MeanVarFC, self).__init__()
        shape = list(input_shape)
        shape[0] = 1
        shape[1] *= 2
        self.param = nn.Parameter(0.01 * torch.randn(shape))

    def forward(self, x):
        x = x + self.param
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 8])]


def get_init_inputs():
    return [[], {'input_shape': [4, 4]}]
