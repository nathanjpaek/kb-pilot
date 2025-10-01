import torch
from torch import nn


class L2ConstrainedLayer(nn.Module):

    def __init__(self, alpha=16):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        l2 = torch.sqrt((x ** 2).sum())
        x = self.alpha * (x / l2)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
