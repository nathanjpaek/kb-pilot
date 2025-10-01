import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):

    def __init__(self, smooth: 'float'=1.0, apply_sigmoid: 'bool'=False):
        super().__init__()
        self.smooth = smooth
        self.apply_sigmoid = apply_sigmoid

    def forward(self, y_pred: 'torch.Tensor', y_true: 'torch.Tensor'
        ) ->torch.Tensor:
        assert y_pred.size() == y_true.size()
        if self.apply_sigmoid:
            y_pred = torch.sigmoid(y_pred)
        y_pred = y_pred.contiguous().view(-1)
        y_true = y_true.contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2.0 * intersection + self.smooth) / (y_pred.sum() + y_true.
            sum() + self.smooth)
        return 1.0 - dsc


class DiceBCELoss(nn.Module):

    def __init__(self, smooth: 'float'=1.0, apply_sigmoid: 'bool'=False):
        super().__init__()
        self.dice_loss = DiceLoss(smooth)
        self.apply_sigmoid = apply_sigmoid

    def forward(self, y_pred: 'torch.Tensor', y_true: 'torch.Tensor'
        ) ->torch.Tensor:
        if self.apply_sigmoid:
            y_pred = torch.sigmoid(y_pred)
        dl = self.dice_loss(y_pred, y_true)
        bce = F.binary_cross_entropy(y_pred, y_true, reduction='mean')
        return dl + bce


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
