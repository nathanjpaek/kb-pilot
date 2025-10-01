import torch
from torch import nn


class ContrastiveLoss(nn.Module):

    def __init__(self, margin=1.0, reduction='mean'):
        super().__init__()
        self.m = margin
        assert reduction in ['mean', 'sum', 'none']
        self.reduction = reduction

    def forward(self, dist, class_):
        dist = dist.transpose(0, -1)
        loss = 0.5 * class_ * dist ** 2 + (1 - class_) * 0.5 * torch.clamp(
            self.m - dist, min=0) ** 2
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
