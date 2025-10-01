import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.nn.functional as F


class MarginRankingLoss(nn.Module):

    def __init__(self, margin=0.2, loss_weight=5e-05, size_average=None,
        reduce=None, reduction='mean'):
        super(MarginRankingLoss, self).__init__()
        self.margin = margin
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, input1, input2, target):
        return self.loss_weight * F.margin_ranking_loss(input1, input2,
            target, margin=self.margin, reduction=self.reduction)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
