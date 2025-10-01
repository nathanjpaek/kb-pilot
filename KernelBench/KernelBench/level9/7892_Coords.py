import torch
import torch.nn as nn
import torch.utils.data
import torch.random


class Coords(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """ adds 2 channels that carry co-ordinate information """
        b, h, w = x.size(0), x.size(2), x.size(3)
        hm = torch.linspace(0, 1, h, dtype=x.dtype, device=x.device).reshape(
            1, 1, h, 1).repeat(b, 1, 1, w)
        wm = torch.linspace(0, 1, w, dtype=x.dtype, device=x.device).reshape(
            1, 1, 1, w).repeat(b, 1, h, 1)
        return torch.cat((x, hm, wm), dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
