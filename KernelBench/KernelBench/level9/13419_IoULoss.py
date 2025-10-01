import torch
import torch.nn as nn
import torch.nn.functional as F


class IoULoss(nn.Module):
    """
    The IoU metric, or Jaccard Index, is similar to the Dice metric and is calculated as the ratio between the overlap 
    of the positive instances between two sets, and their mutual combined values
    """

    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection
        IoU = (intersection + smooth) / (union + smooth)
        return 1 - IoU


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
