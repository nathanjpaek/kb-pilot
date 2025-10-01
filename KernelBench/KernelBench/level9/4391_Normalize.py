import torch
import torch.nn as nn
import torch.nn.functional as functional


class Normalize(nn.Module):

    def __init__(self, dim: 'int', p: 'int'):
        super().__init__()
        self.dim = dim
        self.p = p

    def forward(self, inputs):
        outputs = functional.normalize(inputs, dim=self.dim, p=self.p)
        return outputs


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4, 'p': 4}]
