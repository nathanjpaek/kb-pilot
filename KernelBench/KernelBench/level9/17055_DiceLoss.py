import torch
from torch import nn


class DiceLoss(nn.Module):

    def __init__(self, eps=1e-07):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        loss = 1 - 2.0 * intersection / (pred.sum() + target.sum() + self.eps)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
