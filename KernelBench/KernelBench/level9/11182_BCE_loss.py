import torch
import torch.nn as nn


class BCE_loss(nn.Module):

    def __init__(self):
        super(BCE_loss, self).__init__()

    def forward(self, pred, gt):
        bce_loss = nn.BCELoss(size_average=True)
        bce_out = bce_loss(pred, gt)
        return bce_out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
