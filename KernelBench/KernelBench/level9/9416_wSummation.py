import torch
import torch.nn as nn


class wSummation(nn.Module):
    """
    The spatial weighted summation layer.
    """

    def __init__(self, input_dim):
        """
        :param input_dim: input dimension [C,H,W]
        """
        super(wSummation, self).__init__()
        self.Q = nn.Parameter(torch.rand(input_dim))
        self.Q.requires_grad = True

    def forward(self, x1, x2):
        """
        Calculate the weighted summation of 2 inputs.
        :param x1: input 1
        :param x2: input 2
        :return: the weighted summation
        """
        return x1 * self.Q + (1 - self.Q) * x2


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4}]
