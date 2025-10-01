import torch
import torch.nn as nn


class FocalLoss(nn.Module):

    def __init__(self, gamma=1.5, alpha=0.25, reduction=torch.mean):
        super(FocalLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, true):
        true = true
        pred = pred
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor
        return self.reduction(loss)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
