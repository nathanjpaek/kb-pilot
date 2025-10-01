import torch
import torch.nn as nn


class FocalLoss2d(nn.Module):

    def __init__(self, alpha=0.25, gamma=2, ignore_index=None, reduction=
        'mean', **kwargs):
        super(FocalLoss2d, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-06
        self.ignore_index = ignore_index
        self.reduction = reduction
        assert self.reduction in ['none', 'mean', 'sum']

    def forward(self, prob, target):
        prob = torch.clamp(prob, self.smooth, 1.0 - self.smooth)
        valid_mask = None
        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()
        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()
        if valid_mask is not None:
            pos_mask = pos_mask * valid_mask
            neg_mask = neg_mask * valid_mask
        pos_weight = (pos_mask * torch.pow(1 - prob, self.gamma)).detach()
        pos_loss = -self.alpha * (pos_weight * torch.log(prob))
        neg_weight = (neg_mask * torch.pow(prob, self.gamma)).detach()
        neg_loss = -(1 - self.alpha) * (neg_weight * torch.log(1 - prob))
        loss = pos_loss + neg_loss
        return loss.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
