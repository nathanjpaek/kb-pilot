import torch
import torch.nn as nn


def IoU(logit, truth, smooth=1):
    prob = torch.sigmoid(logit)
    intersection = torch.sum(prob * truth)
    union = torch.sum(prob + truth)
    iou = (2 * intersection + smooth) / (union + smooth)
    return iou


class DiceLoss(nn.Module):

    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logit, truth):
        iou = IoU(logit, truth, self.smooth)
        loss = 1 - iou
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
