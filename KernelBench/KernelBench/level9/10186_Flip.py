import torch
import torch.nn as nn


class Flip(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        xf = torch.flip(x, [2])
        y1 = xf[:, :, 0::2, :]
        y2 = xf[:, :, 1::2, :]
        y = torch.cat((y1, y2), dim=2)
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
