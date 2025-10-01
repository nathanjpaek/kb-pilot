import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional
from typing import no_type_check


class BasicModel_ConvNet_One_Conv(nn.Module):

    def __init__(self, inplace: 'bool'=False):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 3, 1)
        self.relu1 = nn.ReLU(inplace=inplace)
        self.fc1 = nn.Linear(8, 4)
        self.conv1.weight = nn.Parameter(torch.ones(2, 1, 3, 3))
        self.conv1.bias = nn.Parameter(torch.tensor([-50.0, -75.0]))
        self.fc1.weight = nn.Parameter(torch.cat([torch.ones(4, 5), -1 *
            torch.ones(4, 3)], dim=1))
        self.fc1.bias = nn.Parameter(torch.zeros(4))
        self.relu2 = nn.ReLU(inplace=inplace)

    @no_type_check
    def forward(self, x: 'Tensor', x2: 'Optional[Tensor]'=None):
        if x2 is not None:
            x = x + x2
        x = self.relu1(self.conv1(x))
        x = x.view(-1, 8)
        return self.relu2(self.fc1(x))


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
