import torch
from typing import *
import torch.nn as nn


def dice_coeff(input, target, smooth=1.0):
    input_flat = input.view(-1)
    target_flat = target.view(-1)
    intersection = (input_flat * target_flat).sum()
    return (2.0 * intersection + smooth) / (input_flat.sum() + target_flat.
        sum() + smooth)


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        dcoeff = dice_coeff(input, target)
        return 1 - dcoeff


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
