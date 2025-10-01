import torch
import torch.nn as nn


class rotate(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x90 = x.transpose(2, 3).flip(3)
        x180 = x.flip(2).flip(3)
        x270 = x.transpose(2, 3).flip(2)
        x = torch.cat((x, x90, x180, x270), dim=0)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
