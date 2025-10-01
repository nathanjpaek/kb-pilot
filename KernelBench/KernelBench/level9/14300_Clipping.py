import torch
from torch import nn


class Clipping(nn.Module):

    def __init__(self):
        """
        This module perform element-wise clipping.
        """
        super(Clipping, self).__init__()

    def forward(self, x, max_value, min_value):
        """
        The forward function of the clipping module

        :param x:  Input tensor to be clipped
        :param max_value: The maximal value of the tensor after clipping
        :param min_value: The minimal value of the tensor after clipping
        :return: A clipped tensor
        """
        x = torch.min(x, max_value)
        x = torch.max(x, min_value)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
