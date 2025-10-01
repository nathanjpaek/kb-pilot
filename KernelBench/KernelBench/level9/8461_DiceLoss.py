import torch
import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def dice_coef(self, y_pred, y_true):
        pred_probs = torch.sigmoid(y_pred)
        y_true_f = y_true.view(-1)
        y_pred_f = pred_probs.view(-1)
        intersection = torch.sum(y_true_f * y_pred_f)
        return (2.0 * intersection + self.smooth) / (torch.sum(y_true_f) +
            torch.sum(y_pred_f) + self.smooth)

    def forward(self, y_pred, y_true):
        return -self.dice_coef(y_pred, y_true)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
