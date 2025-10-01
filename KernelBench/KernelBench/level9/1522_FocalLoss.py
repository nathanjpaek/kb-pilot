import torch
import torch.nn as nn


class FocalLoss(nn.Module):

    def __init__(self, alpha: 'float'=0.25, gamma: 'float'=2.0, reduction:
        'str'='mean'):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.alpha = alpha
        self.gamma = gamma
        if reduction == 'mean':
            self.reduction = torch.mean
        else:
            self.reduction = torch.sum

    def forward(self, preds, targets):
        bce_loss = self.bce(preds, targets)
        probas = torch.sigmoid(preds)
        loss = torch.where(targets >= 0.5, self.alpha * (1.0 - probas) **
            self.gamma * bce_loss, probas ** self.gamma * bce_loss)
        loss = self.reduction(loss)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
