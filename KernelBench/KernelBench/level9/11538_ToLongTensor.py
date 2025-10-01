import torch
from torch import Tensor
from typing import List
import torch.nn as nn


class ToLongTensor(nn.Module):
    """Convert a list of integers to long tensor
    """

    def __init__(self):
        super(ToLongTensor, self).__init__()

    def forward(self, tokens: 'List[List[int]]') ->Tensor:
        return torch.tensor(tokens)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
