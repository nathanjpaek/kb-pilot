import torch
from typing import *
import torch.nn as nn


class CauchyLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        err = torch.sum(torch.pow(x - y, 2), dim=-1)
        return torch.mean(torch.log(1 + err), dim=-1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
