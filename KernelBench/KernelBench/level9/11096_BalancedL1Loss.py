import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


def balanced_l1_loss(pred, target, beta=1.0, alpha=0.5, gamma=1.5,
    reduction='none'):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    b = np.e ** (gamma / alpha) - 1
    loss = torch.where(diff < beta, alpha / b * (b * diff + 1) * torch.log(
        b * diff / beta + 1) - alpha * diff, gamma * diff + gamma / b - 
        alpha * beta)
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.sum() / pred.numel()
    elif reduction_enum == 2:
        return loss.sum()
    return loss


def weighted_balanced_l1_loss(pred, target, weight, beta=1.0, alpha=0.5,
    gamma=1.5, avg_factor=None):
    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() / 4 + 1e-06
    loss = balanced_l1_loss(pred, target, beta, alpha, gamma, reduction='none')
    return torch.sum(loss * weight)[None] / avg_factor


class BalancedL1Loss(nn.Module):
    """Balanced L1 Loss

    arXiv: https://arxiv.org/pdf/1904.02701.pdf (CVPR 2019)
    """

    def __init__(self, alpha=0.5, gamma=1.5, beta=1.0, loss_weight=1.0):
        super(BalancedL1Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight, *args, **kwargs):
        loss_bbox = self.loss_weight * weighted_balanced_l1_loss(pred,
            target, weight, *args, alpha=self.alpha, gamma=self.gamma, beta
            =self.beta, **kwargs)
        return loss_bbox


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
