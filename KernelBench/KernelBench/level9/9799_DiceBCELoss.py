import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):

    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, predicted, target):
        batch = predicted.size()[0]
        batch_loss = 0
        smooth = 1
        for index in range(batch):
            pre = predicted[index]
            tar = target[index]
            intersection = torch.mul(pre, tar).sum()
            coefficient = (2 * intersection + smooth) / (pre.sum() + tar.
                sum() + smooth)
            batch_loss += coefficient
        batch_loss = batch_loss / batch
        BCE = F.binary_cross_entropy(predicted, target)
        Dice_BCE = BCE + (1 - batch_loss)
        return Dice_BCE


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
