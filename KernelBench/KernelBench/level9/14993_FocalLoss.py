import torch
from torch import nn


class FocalLoss(nn.Module):

    def __init__(self, alpha=0.5, gamma=2, reduction='mean'):
        """FocalLoss
        聚焦损失, 不确定的情况下alpha==0.5效果可能会好一点
        url: https://github.com/CoinCheung/pytorch-loss
        Usage is same as nn.BCEWithLogits:
          >>> loss = criteria(logits, lbs)
        """
        super(FocalLoss, self).__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, labels):
        probs = torch.sigmoid(logits)
        coeff = torch.abs(labels - probs).pow(self.gamma).neg()
        log_0_probs = torch.where(logits >= 0, -logits + nn.functional.
            softplus(logits, -1, 50), -nn.functional.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0, nn.functional.softplus(
            logits, -1, 50), logits - nn.functional.softplus(logits, 1, 50))
        loss = labels * self.alpha * log_1_probs + (1.0 - labels) * (1.0 -
            self.alpha) * log_0_probs
        loss = loss * coeff
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
