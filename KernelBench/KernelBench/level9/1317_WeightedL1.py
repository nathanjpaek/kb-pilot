import torch
import torch.nn as nn


class WeightedL1(nn.Module):

    def __init__(self):
        super(WeightedL1, self).__init__()

    def forward(self, x, target, w):
        return (w * torch.abs(x - target)).mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
