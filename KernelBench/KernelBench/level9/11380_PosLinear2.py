import torch
from torch import Tensor
from torch.utils.data import Dataset as Dataset
import torch.nn as nn
import torch.utils.data


class PosLinear2(torch.nn.Linear):

    def forward(self, x: 'Tensor') ->Tensor:
        return nn.functional.linear(x, torch.nn.functional.softmax(self.
            weight, 1), self.bias)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
