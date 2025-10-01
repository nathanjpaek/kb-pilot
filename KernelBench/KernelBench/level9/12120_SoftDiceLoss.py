import torch
import torch.nn.functional as F
import torch.nn as nn


class SoftDiceLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        """
            Imlements Dice loss function (using Sørensen–Dice coefficient).
        """
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = m1 * m2
        score = (2.0 * intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)
        score = 1 - score.sum() / num
        return score


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
