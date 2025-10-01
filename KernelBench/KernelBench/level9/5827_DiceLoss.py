import torch
import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        smooth = 1e-05
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = input * target
        dice = (2.0 * intersection.sum(1) + smooth) / (input.sum(1) +
            target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return dice


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
