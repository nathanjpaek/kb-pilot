import torch
import torch.nn as nn


class WineLoss(nn.Module):

    def __init__(self):
        super(WineLoss, self).__init__()
        self.smoothl1 = nn.SmoothL1Loss()

    def forward(self, pred, label):
        loss = self.smoothl1(pred, label)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
