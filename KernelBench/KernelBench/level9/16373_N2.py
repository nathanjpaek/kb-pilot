import torch
from typing import Tuple
from abc import ABC
from abc import abstractmethod
from torch import nn


class Regularizer(nn.Module, ABC):

    @abstractmethod
    def forward(self, factors: 'Tuple[torch.Tensor]'):
        pass


class N2(Regularizer):

    def __init__(self, weight: 'float'):
        super(N2, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(torch.norm(f, 2, 1) ** 3)
        return norm / factors[0].shape[0]


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'weight': 4}]
