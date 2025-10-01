import torch
from torch import nn


class DiceLoss(nn.Module):
    """
    DICE loss function

    Args:
        alpha (default: int=10): Coefficient in exp of sigmoid function  
        smooth (default: int=1): To prevent zero in nominator
    """

    def __init__(self, alpha=10, smooth=1):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth

    def sigmoid(self, x):
        return 1.0 / (1.0 + torch.exp(-self.alpha * x))

    def forward(self, fake, real):
        fake = self.sigmoid(fake)
        intersection = (fake * real).sum() + self.smooth
        union = fake.sum() + real.sum() + self.smooth
        dice = torch.div(2 * intersection, union)
        loss = 1.0 - dice
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
