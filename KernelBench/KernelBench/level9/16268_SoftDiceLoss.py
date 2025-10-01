import torch
import torch.nn as nn


class SoftDiceLoss(nn.Module):

    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, output, label):
        probs = output.view(-1)
        mask = label.view(-1)
        smooth = 1
        intersection = torch.sum(probs * mask)
        den1 = torch.sum(probs)
        den2 = torch.sum(mask)
        soft_dice = (2 * intersection + smooth) / (den1 + den2 + smooth)
        return -soft_dice


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
