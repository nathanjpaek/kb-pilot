import torch
import torch.nn as nn
from typing import Any
import torch.nn.functional as F


class BalancedBinaryCrossEntropy(nn.Module):
    """二分类加权交叉熵"""

    def __init__(self, reduction: 'str'='mean', class_weight: 'Any'=None,
        loss_weight: 'float'=1.0, activation: 'bool'=False) ->None:
        super(BalancedBinaryCrossEntropy, self).__init__()
        self.reduction = reduction
        self.class_weight = class_weight
        self.loss_weight = loss_weight
        self.activation = activation

    def forward(self, pred, target):
        assert self.reduction in ['sum', 'mean'
            ], "reduction is in ['sum', 'mean']"
        if self.activation:
            pred = torch.sigmoid(pred)
        assert pred.size(0) == target.size(0)
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)
        if self.class_weight is not None:
            class_weight = self.class_weight[0] * target + self.class_weight[1
                ] * (1 - target)
            self.class_weight = class_weight.clone().detach().requires_grad_(
                False)
        loss = F.binary_cross_entropy(pred, target, reduction='none',
            weight=self.class_weight)
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
