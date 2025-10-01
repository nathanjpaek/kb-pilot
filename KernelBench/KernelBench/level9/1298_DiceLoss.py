import torch
from torch import nn


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, predict, target):
        N = target.size(0)
        smooth = 1
        predict_flat = predict.view(N, -1)
        target_flat = target.view(N, -1)
        intersection = (predict_flat * target_flat).sum(1)
        union = predict_flat.sum(1) + target_flat.sum(1)
        loss = (2 * intersection + smooth) / (union + smooth)
        loss = 1 - loss.sum() / N
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
