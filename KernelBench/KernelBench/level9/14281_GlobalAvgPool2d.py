import torch
from torch import nn
from torch.nn import functional as F


class GlobalAvgPool2d(nn.Module):

    def __init__(self):
        """
        Global Average pooling module
        """
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        """
        The forward function of the GlobalAvgPool2d module

        :param x: Input tensor x
        :return: A tensor after average pooling
        """
        return F.avg_pool2d(x, (x.shape[2], x.shape[3]))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
