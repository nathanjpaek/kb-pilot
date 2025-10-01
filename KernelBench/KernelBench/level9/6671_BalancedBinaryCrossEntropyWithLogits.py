import torch
import torch.nn as nn
from typing import Any


class BalancedBinaryCrossEntropyWithLogits(nn.Module):
    """二分类加权交叉熵"""

    def __init__(self, reduction: 'str'='mean', class_weight: 'Any'=None,
        loss_weight: 'float'=1.0, activation: 'bool'=False, eposion:
        'float'=1e-10) ->None:
        super(BalancedBinaryCrossEntropyWithLogits, self).__init__()
        self.reduction = reduction
        self.class_weight = class_weight
        self.loss_weight = loss_weight
        self.eps = eposion

    def forward(self, pred, target):
        assert self.reduction in ['sum', 'mean'
            ], "reduction is in ['sum', 'mean']"
        assert pred.size(0) == target.size(0)
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)
        count_pos = torch.sum(target) * 1.0 + self.eps
        count_neg = torch.sum(1.0 - target) * 1.0
        beta = count_neg / count_pos
        beta_back = count_pos / (count_pos + count_neg)
        bce1 = nn.BCEWithLogitsLoss(pos_weight=beta, reduction='none')
        loss = beta_back * bce1(pred, target)
        loss = loss.mean(dim=1)
        if self.reduction == 'mean':
            loss = loss.mean(dim=0)
        else:
            loss = loss.sum(dim=0)
        return self.loss_weight * loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
