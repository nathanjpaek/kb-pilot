import torch
import torch.nn as nn


class FocalDiceLoss(nn.Module):

    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, score, target):
        target = target.float()
        smooth = 1e-06
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        dice = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - dice ** (1.0 / self.gamma)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
