import torch
from torch import nn


class Normalization(nn.Module):

    def __init__(self, mean=torch.zeros(3), std=torch.ones(3)):
        super(Normalization, self).__init__()
        self.mean = nn.Parameter(mean.view(-1, 1, 1), requires_grad=False)
        self.std = nn.Parameter(std.view(-1, 1, 1), requires_grad=False)

    def forward(self, img):
        return (img - self.mean) / self.std


def get_inputs():
    return [torch.rand([4, 3, 4, 4])]


def get_init_inputs():
    return [[], {}]
