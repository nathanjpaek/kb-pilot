import torch
import torch.nn as nn
import torch.utils.data
import torch


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        pred = pred.squeeze(dim=1)
        dice = 2 * (pred * target).sum(dim=1).sum(dim=1).sum(dim=1) / (pred
            .pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + target.pow(2).sum(
            dim=1).sum(dim=1).sum(dim=1) + 1e-05)
        return (1 - dice).mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
