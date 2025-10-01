import torch
import torch.nn as nn
from typing import *


class DiceLoss(nn.Module):

    def __init__(self, smooth: 'float'=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num = targets.size(0)
        probs = torch.sigmoid(logits)
        m1, m2 = probs.view(num, -1), targets.view(num, -1)
        intersection = m1 * m2
        score = 2.0 * (intersection.sum(1) + self.smooth) / (m1.sum(1) + m2
            .sum(1) + self.smooth)
        score = 1 - score.sum() / num
        return score


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
