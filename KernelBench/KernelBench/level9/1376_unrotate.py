import torch
import torch.nn as nn


class unrotate(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x0, x90, x180, x270 = torch.chunk(x, 4, dim=0)
        x90 = x90.transpose(2, 3).flip(2)
        x180 = x180.flip(2).flip(3)
        x270 = x270.transpose(2, 3).flip(3)
        x = torch.cat((x0, x90, x180, x270), dim=1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
