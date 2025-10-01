import torch
import torch.nn as nn


class SmoothL1Loss(nn.Module):

    def __init__(self, beta=1.0, reduction='mean'):
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, pred, target, weight=None):
        assert pred.size() == target.size() and target.numel() > 0
        diff = torch.abs(pred - target)
        loss = torch.where(diff < self.beta, 0.5 * diff * diff / self.beta,
            diff - 0.5 * self.beta)
        if weight is not None:
            loss = loss * weight
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
