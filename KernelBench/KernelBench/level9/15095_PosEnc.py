import torch
import torch.nn as nn
import torch.utils.data
import torch.utils
from matplotlib import cm as cm
from torch.nn.parallel import *
from torchvision.models import *
from torchvision.datasets import *


class PosEnc(nn.Module):

    def __init__(self, C, ks):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, C, ks, ks))

    def forward(self, x):
        return x + self.weight


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'C': 4, 'ks': 4}]
