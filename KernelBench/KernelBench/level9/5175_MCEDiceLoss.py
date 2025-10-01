import torch
import torch.nn.functional
import torch.nn as nn


def centercrop(image, w, h):
    _nt, _ct, ht, wt = image.size()
    padw, padh = (wt - w) // 2, (ht - h) // 2
    if padw > 0 and padh > 0:
        image = image[:, :, padh:-padh, padw:-padw]
    return image


class BCELoss(nn.Module):

    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_true):
        _n, _ch, h, w = y_pred.size()
        y_true = centercrop(y_true, w, h)
        loss = self.bce(y_pred, y_true)
        return loss


class BLogDiceLoss(nn.Module):

    def __init__(self, classe=1):
        super(BLogDiceLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classe = classe

    def forward(self, y_pred, y_true, weight=None):
        _n, _ch, h, w = y_pred.size()
        y_true = centercrop(y_true, w, h)
        y_pred = self.sigmoid(y_pred)
        eps = 1e-15
        dice_target = (y_true[:, self.classe, ...] == 1).float()
        dice_output = y_pred[:, self.classe, ...]
        intersection = (dice_output * dice_target).sum()
        union = dice_output.sum() + dice_target.sum() + eps
        return -torch.log(2 * intersection / union)


class MCEDiceLoss(nn.Module):

    def __init__(self, alpha=1.0, gamma=1.0):
        super(MCEDiceLoss, self).__init__()
        self.loss_mce = BCELoss()
        self.loss_dice = BLogDiceLoss(classe=1)
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true, weight=None):
        loss_all = self.loss_mce(y_pred[:, :2, ...], y_true[:, :2, ...])
        loss_fg = self.loss_dice(y_pred, y_true)
        loss = loss_all + 2.0 * loss_fg
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
