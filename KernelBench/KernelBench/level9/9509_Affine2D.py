import torch
import torch.nn as nn


class Affine2D(nn.Module):

    def __init__(self, cin):
        """

        :param cin:
        """
        super(Affine2D, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, cin, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, cin, 1, 1))

    def forward(self, x):
        """

        :param x:
        :return:
        """
        return self.weight * x + self.bias


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'cin': 4}]
