import torch
import torch.nn as nn


class RevPaddingLayer(nn.Module):

    def __init__(self, stride):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)

    def forward(self, x):
        x = self.pool(x)
        zeros = torch.zeros_like(x)
        zeros_left, zeros_right = zeros.chunk(2, dim=1)
        y = torch.cat([zeros_left, x, zeros_right], dim=1)
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'stride': 1}]
