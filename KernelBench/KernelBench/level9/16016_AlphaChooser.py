import torch
from torch import nn


class AlphaChooser(torch.nn.Module):
    """
    It manages the alpha values in alpha-entmax
    function.
    """

    def __init__(self, head_count):
        super(AlphaChooser, self).__init__()
        self.pre_alpha = nn.Parameter(torch.randn(head_count))

    def forward(self):
        alpha = 1 + torch.sigmoid(self.pre_alpha)
        return torch.clamp(alpha, min=1.01, max=2)


def get_inputs():
    return []


def get_init_inputs():
    return [[], {'head_count': 4}]
