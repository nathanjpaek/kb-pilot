import torch
from torch import nn


class BCEDiceLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, input, target):
        pred = input.view(-1)
        truth = target.view(-1)
        bce_loss = nn.BCELoss()(pred, truth).double()
        dice_coef = (2.0 * (pred * truth).double().sum() + 1) / (pred.
            double().sum() + truth.double().sum() + 1)
        return bce_loss + (1 - dice_coef)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
