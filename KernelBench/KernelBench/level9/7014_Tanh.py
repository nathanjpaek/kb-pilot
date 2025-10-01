import torch
import torch.utils.data
import torch.nn as nn
import torch._utils
from torch import optim as optim
import torch.nn.parallel


class Tanh(nn.Module):

    def __init__(self, inplace=False):
        super(Tanh, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.tanh_() if self.inplace else x.tanh()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
