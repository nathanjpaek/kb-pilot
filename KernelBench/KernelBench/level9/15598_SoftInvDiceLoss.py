import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftInvDiceLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super(SoftInvDiceLoss, self).__init__()

    def forward(self, logits, targets):
        smooth = 1.0
        logits = F.sigmoid(logits)
        iflat = 1 - logits.view(-1)
        tflat = 1 - targets.view(-1)
        intersection = (iflat * tflat).sum()
        return 1 - (2.0 * intersection + smooth) / (iflat.sum() + tflat.sum
            () + smooth)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
