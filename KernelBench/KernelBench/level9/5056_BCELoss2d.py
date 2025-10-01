import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss2d(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super(BCELoss2d, self).__init__()
        self.criterion = nn.BCELoss(weight, size_average)

    def forward(self, inputs, targets):
        probs = F.sigmoid(inputs)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        loss = self.criterion(probs_flat, targets_flat)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
