import torch
import torch.nn as nn


class RBF(nn.Module):

    def __init__(self):
        super(RBF, self).__init__()
        self.mean = nn.Parameter(torch.Tensor([0.0]))
        self.std = nn.Parameter(torch.Tensor([1.0]))

    def forward(self, x):
        gauss = torch.exp(-(x - self.mean) ** 2 / (2 * self.std ** 2))
        return gauss


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
