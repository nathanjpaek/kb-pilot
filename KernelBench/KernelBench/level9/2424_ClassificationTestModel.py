from torch.nn import Module
import torch
import torch.nn as nn
from typing import Any
from torch.nn.modules import Module


class ClassificationTestModel(Module):

    def __init__(self, in_chans: 'int'=3, num_classes: 'int'=1000, **kwargs:
        Any) ->None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_chans, out_channels=1,
            kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1, num_classes)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = self.conv1(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
