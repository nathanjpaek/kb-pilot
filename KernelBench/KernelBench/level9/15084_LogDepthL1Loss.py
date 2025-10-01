import torch
import torch.nn as nn


class LogDepthL1Loss(nn.Module):

    def __init__(self, eps=1e-05):
        super(LogDepthL1Loss, self).__init__()
        self.eps = eps

    def forward(self, pred, gt):
        pred = pred.view(-1)
        gt = gt.view(-1)
        mask = gt > self.eps
        diff = torch.abs(torch.log(gt[mask]) - pred[mask])
        return diff.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
