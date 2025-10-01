import torch
import torch.nn as nn


class SoftDiceLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets):
        smooth = 1.0
        num = targets.size(0)
        m1 = inputs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = m1 * m2
        score = (2.0 * intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(
            1) + smooth)
        dice = score.sum() / num
        return 1.0 - dice


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
