import torch
import numpy as np
from torch import nn
from typing import *


class AlphaVectorMultiplication(nn.Module):

    def __init__(self, size_alpha):
        super(AlphaVectorMultiplication, self).__init__()
        self.size_alpha = size_alpha
        self.alpha = nn.Parameter(torch.from_numpy(np.zeros((1, size_alpha),
            np.float32)))

    def forward(self, x):
        bsz = x.size()[0]
        x = x * torch.sigmoid(self.alpha.expand(bsz, -1))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'size_alpha': 4}]
