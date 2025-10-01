import torch
import torch.nn as nn


class SoftDiceLoss(nn.Module):
    """
    Soft Dice Loss
    """

    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        smooth = 1.0
        logits = torch.sigmoid(logits)
        iflat = logits.view(-1)
        tflat = targets.view(-1)
        intersection = (iflat * tflat).sum()
        return 1 - (2.0 * intersection + smooth) / (iflat.sum() + tflat.sum
            () + smooth)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
