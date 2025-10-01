import torch
import torch.nn as nn


class DetNet2(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, 1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        return x


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
