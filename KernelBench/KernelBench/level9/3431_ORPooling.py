import torch
import torch.nn as nn


class ORPooling(nn.Module):

    def __init__(self, orientations):
        super(ORPooling, self).__init__()
        self.orientations = orientations

    def forward(self, x):
        B, C, H, W = x.shape
        assert C % self.orientations == 0
        x = x.view(B, -1, self.orientations, H, W)
        return x.max(2)[0]


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'orientations': 4}]
