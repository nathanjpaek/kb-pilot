import torch
from torch import nn


class WassersteinLoss(nn.Module):
    """For WGAN."""

    def forward(self, real, fake):
        return real.mean() - fake.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
