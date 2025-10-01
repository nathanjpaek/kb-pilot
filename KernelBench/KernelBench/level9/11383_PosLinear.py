import torch
from torch import Tensor
from torch.utils.data import Dataset as Dataset
import torch.nn as nn
import torch.utils.data


class PosLinear(torch.nn.Linear):

    def forward(self, x: 'Tensor') ->Tensor:
        gain = 1 / x.size(1)
        return nn.functional.linear(x, torch.nn.functional.softplus(self.
            weight), self.bias) * gain


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
