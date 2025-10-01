import torch
from torch import nn


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target, weight=None):
        smooth = 1
        size = pred.size(0)
        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)
        intersection = pred_flat * target_flat
        dice_score = (2 * intersection.sum(1) + smooth) / (pred_flat.sum(1) +
            target_flat.sum(1) + smooth)
        if weight is not None:
            dice_score = weight * dice_score
            dice_loss = weight.sum() / size - dice_score.sum() / size
        else:
            dice_loss = 1 - dice_score.sum() / size
        return dice_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
