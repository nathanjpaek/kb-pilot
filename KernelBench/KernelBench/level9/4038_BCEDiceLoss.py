import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class BCEDiceLoss(nn.Module):

    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-05
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = input * target
        dice = (2.0 * intersection.sum(1) + smooth) / (input.sum(1) +
            target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + 0.5 * dice


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
