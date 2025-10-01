import torch
import torch.nn as nn
import torch._C
import torch.serialization


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 1)

    def forward(self, x):
        return self.conv(x)


def get_inputs():
    return [torch.rand([4, 2, 64, 64])]


def get_init_inputs():
    return [[], {}]
