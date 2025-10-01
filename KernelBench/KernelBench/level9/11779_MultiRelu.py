import torch
from torch import Tensor
from typing import Tuple
import torch.nn as nn
from typing import no_type_check


class MultiRelu(nn.Module):

    def __init__(self, inplace: 'bool'=False) ->None:
        super().__init__()
        self.relu1 = nn.ReLU(inplace=inplace)
        self.relu2 = nn.ReLU(inplace=inplace)

    @no_type_check
    def forward(self, arg1: 'Tensor', arg2: 'Tensor') ->Tuple[Tensor, Tensor]:
        return self.relu1(arg1), self.relu2(arg2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
