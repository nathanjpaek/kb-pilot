import torch
from torch import nn


class HSwish(nn.Module):

    def __init__(self):
        """
        An HSwish module
        :param inplace: A boolean stating if the operation is inplace
        """
        super(HSwish, self).__init__()
        self.relu6 = nn.ReLU6()

    def forward(self, x):
        """
        The forward function of the HSwish module
        :param x: Input tensor x
        :return: A tensor after HSwish
        """
        return x * self.relu6(x + 3.0) / 6.0


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
