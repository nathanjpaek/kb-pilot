import torch
import numpy as np
import torch.nn as nn
import torch.onnx


def balanced_l1_loss(pred, target, beta=1.0, alpha=0.5, gamma=1.5,
    reduction='none'):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    b = np.e ** (gamma / alpha) - 1
    loss = torch.where(diff < beta, alpha / b * (b * diff + 1) * torch.log(
        b * diff / beta + 1) - alpha * diff, gamma * diff + gamma / b - 
        alpha * beta)
    if reduction == 'none':
        loss = loss
    elif reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'mean':
        loss = loss.mean()
    else:
        raise NotImplementedError
    return loss


class BalancedL1Loss(nn.Module):
    """Balanced L1 Loss
    arXiv: https://arxiv.org/pdf/1904.02701.pdf (CVPR 2019)
    """

    def __init__(self, loss_weight=1.0, beta=1.0, alpha=0.5, gamma=1.5,
        reduction='none'):
        super(BalancedL1Loss, self).__init__()
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.weight = loss_weight
        self.reduction = reduction
        assert reduction == 'none', 'only none reduction is support!'

    def forward(self, pred, gt, mask=None):
        num = pred.size(0) * pred.size(1)
        if mask is not None:
            num = mask.float().sum()
            mask = mask.unsqueeze(2).expand_as(pred).float()
            pred = pred * mask
            gt = gt * mask
        loss = balanced_l1_loss(pred, gt, beta=self.beta, alpha=self.alpha,
            gamma=self.gamma, reduction=self.reduction)
        loss = loss.sum() / (num + 0.0001) * self.weight
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
