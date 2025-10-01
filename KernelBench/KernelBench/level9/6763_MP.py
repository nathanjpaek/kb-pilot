from torch.nn import Module
import torch
import torch.utils.data
from torch.nn import MaxPool2d


class MP(Module):

    def __init__(self, k=2):
        super().__init__()
        self.m = MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
