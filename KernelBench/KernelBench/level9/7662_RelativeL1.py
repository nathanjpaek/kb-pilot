import torch
import torch.utils.data
from torch import nn
import torch.jit


class RelativeL1(nn.Module):

    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.L1Loss()

    def forward(self, input, target):
        base = target + 0.01
        return self.criterion(input / base, target / base)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
