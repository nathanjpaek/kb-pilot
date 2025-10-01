from torch.nn import Module
import torch
import torch.nn as nn
from typing import Any
from typing import cast
from torch.nn.modules import Module


class SegmentationTestModel(Module):

    def __init__(self, in_channels: 'int'=3, classes: 'int'=1000, **kwargs: Any
        ) ->None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=
            classes, kernel_size=1, padding=0)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return cast(torch.Tensor, self.conv1(x))


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
