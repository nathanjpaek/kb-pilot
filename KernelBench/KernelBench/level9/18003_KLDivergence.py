import torch
from torch import nn


def kl_divergence(px, py):
    eps = 1e-08
    kl_div = px * (torch.log(px + eps) - torch.log(py + eps))
    return kl_div


class KLDivergence(nn.Module):
    """
    Kullback–Leibler divergence

    Args:
        - None -
    """

    def __init__(self):
        super().__init__()

    def forward(self, px, py):
        return kl_divergence(px, py)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
