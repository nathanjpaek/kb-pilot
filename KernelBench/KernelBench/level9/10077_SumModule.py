import torch
import torch.nn as nn


class SumModule(nn.Module):

    def __init__(self, *axis, keepdim=False):
        super().__init__()
        self.axis = axis
        self.keepdim = keepdim

    def forward(self, v):
        sum = v.sum(self.axis)
        if self.keepdim:
            dims = list(v.shape)
            if isinstance(self.axis, list) or isinstance(self.axis, tuple):
                for ax in self.axis:
                    dims[ax] = 1
            else:
                dims[self.axis] = 1
            sum = sum.view(dims)
        return sum


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
