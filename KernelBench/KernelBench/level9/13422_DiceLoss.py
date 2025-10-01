import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    The Dice coefficient, or Dice-Sørensen coefficient, is a common metric for pixel segmentation
    """

    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, labels, smooth=1):
        inputs = F.sigmoid(inputs)
        inputs = inputs.view(-1)
        labels = labels.view(-1)
        intersection = (inputs * labels).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + labels.sum() +
            smooth)
        return 1 - dice


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
