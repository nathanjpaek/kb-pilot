import torch
from torch import nn
from torch.nn import Sequential
from torch.nn import Conv2d
import torch.nn
import torch.optim


class Lenet(nn.Module):

    def __init__(self):
        super(Lenet, self).__init__()
        layer1 = Sequential()
        layer1.add_module('conv1', Conv2d(3, 32, 3, 1, padding=1))
        self.layer1 = layer1

    def forward(self, x):
        rt = self.layer1(x)
        rt = rt.view(rt.size(0), -1)
        return rt


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
