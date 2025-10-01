import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs: 'torch.Tensor', targets: 'torch.Tensor',
        smooth: 'int'=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (inputs.sum() +
            targets.sum() + smooth)
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction
            ='mean')
        Dice_BCE = BCE + dice_loss
        return Dice_BCE


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
