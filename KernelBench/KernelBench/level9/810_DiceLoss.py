import torch
import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets):
        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + 1e-05) / (inputs.sum() + targets.sum() +
            1e-05)
        return 1 - dice


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
