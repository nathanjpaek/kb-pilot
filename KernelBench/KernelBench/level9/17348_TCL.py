import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from torch.nn.init import *


class TCL(nn.Module):

    def __init__(self, conv_size, dim):
        super(TCL, self).__init__()
        self.conv2d = nn.Conv2d(dim, dim, kernel_size=(conv_size, 1),
            padding=(conv_size // 2, 0))
        kaiming_normal_(self.conv2d.weight)

    def forward(self, x):
        x = self.conv2d(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'conv_size': 4, 'dim': 4}]
