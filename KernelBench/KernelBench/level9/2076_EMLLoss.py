import torch
import torch.nn as nn
from torch import optim as optim


class EMLLoss(nn.Module):

    def __init__(self):
        super(EMLLoss, self).__init__()

    def forward(self, y_pred, y_true):
        gamma = 1.1
        alpha = 0.48
        smooth = 1.0
        epsilon = 1e-07
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)
        intersection = (y_true * y_pred).sum()
        dice_loss = (2.0 * intersection + smooth) / ((y_true * y_true).sum(
            ) + (y_pred * y_pred).sum() + smooth)
        y_pred = torch.clamp(y_pred, epsilon)
        pt_1 = torch.where(torch.eq(y_true, 1), y_pred, torch.ones_like(y_pred)
            )
        pt_0 = torch.where(torch.eq(y_true, 0), y_pred, torch.zeros_like(
            y_pred))
        focal_loss = -torch.mean(alpha * torch.pow(1.0 - pt_1, gamma) *
            torch.log(pt_1)) - torch.mean((1 - alpha) * torch.pow(pt_0,
            gamma) * torch.log(1.0 - pt_0))
        return focal_loss - torch.log(dice_loss)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
