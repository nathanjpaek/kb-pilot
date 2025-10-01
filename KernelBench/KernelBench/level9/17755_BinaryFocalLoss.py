import torch
import torch.nn as nn


def binary_focal_loss(pred, target, gamma=2.0, alpha=-1, reduction='mean'):
    p = torch.sigmoid(pred)
    loss_pos = -target * (1.0 - p) ** gamma * torch.log(p + 1e-09)
    loss_neg = -(1.0 - target) * p ** gamma * torch.log(1.0 - p + 1e-09)
    if alpha >= 0.0 and alpha <= 1.0:
        loss_pos = loss_pos * alpha
        loss_neg = loss_neg * (1.0 - alpha)
    loss = loss_pos + loss_neg
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise RuntimeError


class BinaryFocalLoss(nn.Module):

    def __init__(self, gamma=2.0, alpha=-1):
        super(BinaryFocalLoss, self).__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, pred, target, reduction='mean'):
        return binary_focal_loss(pred, target, self.gamma, self.alpha,
            reduction)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
