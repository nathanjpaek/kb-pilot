import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    def __init__(self, gamma=1, weight=None, balance=0.75):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.balance = balance
        return

    def forward(self, inputs, target):
        logpt = -F.binary_cross_entropy_with_logits(input=inputs, target=
            target, reduction='none')
        if self.weight is not None:
            logpt = logpt * self.weight
        logpt = logpt.mean()
        pt = torch.exp(logpt)
        focal_loss = -(1 - pt) ** self.gamma * logpt
        balanced_focal_loss = self.balance * focal_loss
        return balanced_focal_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
