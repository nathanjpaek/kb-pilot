import torch
from typing import Tuple
from torch import nn
from abc import ABC
from abc import abstractmethod


class Regularizer(nn.Module, ABC):

    @abstractmethod
    def forward(self, factors: 'Tuple[torch.Tensor]'):
        pass


class Lambda3(Regularizer):

    def __init__(self, weight: 'float'):
        super(Lambda3, self).__init__()
        self.weight = weight

    def forward(self, factor):
        ddiff = factor[1:] - factor[:-1]
        rank = int(ddiff.shape[1] / 2)
        diff = torch.sqrt(ddiff[:, :rank] ** 2 + ddiff[:, rank:] ** 2) ** 3
        return self.weight * torch.sum(diff) / (factor.shape[0] - 1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'weight': 4}]
