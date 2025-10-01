import torch
from torch import nn


class PixelwiseLossL1(nn.Module):
    """
    L1 loss function

    Args:
        alpha (default: int=1): Coefficient by which loss will be multiplied
    """

    def __init__(self, alpha=1):
        super().__init__()
        self.alpha = alpha
        self.criterion = nn.L1Loss()

    def forward(self, fake, real):
        return self.alpha * self.criterion(fake, real)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
