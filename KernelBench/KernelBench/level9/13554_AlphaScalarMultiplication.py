import torch
import numpy as np
from torch import nn
from typing import *


class AlphaScalarMultiplication(nn.Module):

    def __init__(self, size_alpha_x, size_alpha_y):
        super(AlphaScalarMultiplication, self).__init__()
        self.size_alpha_x = size_alpha_x
        self.size_alpha_y = size_alpha_y
        self.alpha_x = nn.Parameter(torch.from_numpy(np.zeros(1, np.float32)))

    def forward(self, x, y):
        bsz = x.size()[0]
        factorx = torch.sigmoid(self.alpha_x.expand(bsz, self.size_alpha_x))
        factory = 1.0 - torch.sigmoid(self.alpha_x.expand(bsz, self.
            size_alpha_y))
        x = x * factorx
        y = y * factory
        return x, y


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'size_alpha_x': 4, 'size_alpha_y': 4}]
