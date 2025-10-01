import torch
import torch.nn as nn
import torch.nn.functional as F


class TilePad2d(nn.Module):

    def __init__(self, left, right, top, bottom):
        super().__init__()
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom

    def forward(self, x):
        return F.pad(x, [self.left, self.right, self.top, self.bottom],
            mode='circular')


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'left': 4, 'right': 4, 'top': 4, 'bottom': 4}]
