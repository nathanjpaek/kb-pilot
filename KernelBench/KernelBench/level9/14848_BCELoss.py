import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils


def binary_cross_entropy(inputs, target, weight=None, reduction='mean',
    smooth_eps=None, from_logits=False):
    """cross entropy loss, with support for label smoothing https://arxiv.org/abs/1512.00567"""
    smooth_eps = smooth_eps or 0
    if smooth_eps > 0:
        target = target.float()
        target.add_(smooth_eps).div_(2.0)
    if from_logits:
        return F.binary_cross_entropy_with_logits(inputs, target, weight=
            weight, reduction=reduction)
    else:
        return F.binary_cross_entropy(inputs, target, weight=weight,
            reduction=reduction)


class BCELoss(nn.BCELoss):

    def __init__(self, weight=None, size_average=None, reduce=None,
        reduction='mean', smooth_eps=None, from_logits=False):
        super(BCELoss, self).__init__(weight, size_average, reduce, reduction)
        self.smooth_eps = smooth_eps
        self.from_logits = from_logits

    def forward(self, input, target):
        return binary_cross_entropy(input, target, weight=self.weight,
            reduction=self.reduction, smooth_eps=self.smooth_eps,
            from_logits=self.from_logits)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
