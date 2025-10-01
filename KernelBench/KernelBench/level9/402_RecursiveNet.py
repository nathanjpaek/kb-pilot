import torch
import torch.nn as nn


class RecursiveNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)

    def forward(self, x, args1=None, args2=None):
        del args1, args2
        for _ in range(3):
            out = self.conv1(x)
            out = self.conv1(out)
        return out


def get_inputs():
    return [torch.rand([4, 64, 64, 64])]


def get_init_inputs():
    return [[], {}]
