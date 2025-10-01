import torch
import torch.nn as nn


class L2loss(nn.Module):
    """
    Euclidean loss also known as L2 loss. Compute the sum of the squared difference between the two images.
    """

    def __init__(self):
        super(L2loss, self).__init__()

    def forward(self, input, target):
        return torch.sum((input - target) ** 2) / 2


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
