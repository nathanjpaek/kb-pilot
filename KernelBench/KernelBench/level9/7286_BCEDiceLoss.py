import torch
from torch import nn
from torch import torch


class BCEDiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, Yp, Yt, smooth=1e-07):
        num = Yt.size(0)
        Yp = Yp.view(num, -1)
        Yt = Yt.view(num, -1)
        bce = nn.functional.binary_cross_entropy(Yp, Yt)
        intersection = (Yp * Yt).sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (Yp.sum() + Yt.sum(
            ) + smooth)
        bce_dice_loss = bce + dice_loss
        return bce_dice_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
