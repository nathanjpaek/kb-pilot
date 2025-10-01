import torch
from torch import nn


class PixelwiseLossMSE(nn.Module):
    """
    MSE loss function

    Args:
        alpha (default: int=20): Coefficient by which loss will be multiplied
    """

    def __init__(self, alpha=20):
        super().__init__()
        self.alpha = alpha

    def forward(self, fake, real):
        return self.alpha * torch.mean((fake - real) ** 2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
