import torch
import torch.nn as nn


class PixLoss(nn.Module):
    """Pixel-wise MSE loss for images"""

    def __init__(self, alpha=20):
        super().__init__()
        self.alpha = alpha

    def forward(self, fake, real):
        return self.alpha * torch.mean((fake - real) ** 2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
