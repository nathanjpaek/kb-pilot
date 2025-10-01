import torch
import torch.nn as nn


class SVIGlobalMaxPool2D(nn.Module):
    """
    Expects
    :param x: [examples, samples, channels, H, W]
    :return: [examples, samples, channels]
    """

    def __init__(self):
        super(SVIGlobalMaxPool2D, self).__init__()

    def forward(self, x):
        x = x.max(4)[0].max(3)[0]
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
