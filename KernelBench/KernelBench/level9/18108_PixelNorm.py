import torch
import torch.nn as nn


class PixelNorm(nn.Module):

    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-08

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) +
            self.epsilon)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
