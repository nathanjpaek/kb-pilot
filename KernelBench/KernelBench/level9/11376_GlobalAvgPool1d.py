import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from abc import abstractmethod
from torch.nn import functional
from typing import *


class AvgPool(nn.Module):
    """
    AvgPool Module.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, input_tensor):
        pass


class GlobalAvgPool1d(AvgPool):
    """
    GlobalAvgPool1d Module.
    """

    def forward(self, input_tensor):
        return functional.avg_pool1d(input_tensor, input_tensor.size()[2:]
            ).view(input_tensor.size()[:2])


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {}]
