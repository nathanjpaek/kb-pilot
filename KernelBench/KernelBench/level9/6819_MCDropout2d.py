import torch
from torch import Tensor
import torch.nn as nn
from torch.functional import F
import torch.nn.functional as F


class MCDropout2d(nn.Dropout2d):
    """2D dropout that stays on during training and testing

    """

    def forward(self, input: 'Tensor') ->Tensor:
        return F.dropout2d(input, self.p, True, self.inplace)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
