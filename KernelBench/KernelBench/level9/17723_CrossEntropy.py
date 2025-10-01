import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim


class CrossEntropy(nn.Module):

    def __init__(self, ignore_label=-1, weight=None, reduction='mean'):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=
            ignore_label, reduction=reduction)

    def forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(score, size=(h, w), mode='bilinear',
                align_corners=False)
        loss = self.criterion(score, target)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
