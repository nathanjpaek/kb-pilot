import torch
import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.contiguous()
        targets = targets.contiguous()
        intersection = (inputs * targets).sum(dim=2).sum(dim=2)
        dice = (2.0 * intersection + smooth) / (inputs.sum(dim=2).sum(dim=2
            ) + targets.sum(dim=2).sum(dim=2) + smooth)
        loss = 1 - dice
        return loss.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
