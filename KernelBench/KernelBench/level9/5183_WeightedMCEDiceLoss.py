import torch
import torch.nn.functional
import torch.nn as nn
import torch.nn.functional as F


def centercrop(image, w, h):
    _nt, _ct, ht, wt = image.size()
    padw, padh = (wt - w) // 2, (ht - h) // 2
    if padw > 0 and padh > 0:
        image = image[:, :, padh:-padh, padw:-padw]
    return image


class WeightedMCEFocalloss(nn.Module):

    def __init__(self, gamma=2.0):
        super(WeightedMCEFocalloss, self).__init__()
        self.gamma = gamma

    def forward(self, y_pred, y_true, weight):
        _n, _ch, h, w = y_pred.size()
        y_true = centercrop(y_true, w, h)
        weight = centercrop(weight, w, h)
        y_pred_log = F.log_softmax(y_pred, dim=1)
        fweight = (1 - F.softmax(y_pred, dim=1)) ** self.gamma
        weight = weight * fweight
        logpy = torch.sum(weight * y_pred_log * y_true, dim=1)
        loss = -torch.mean(logpy)
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


class WeightedMCEDiceLoss(nn.Module):

    def __init__(self, alpha=1.0, gamma=1.0):
        super(WeightedMCEDiceLoss, self).__init__()
        self.loss_mce = WeightedMCEFocalloss()
        self.loss_dice = BLogDiceLoss()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true, weight):
        alpha = self.alpha
        weight = torch.pow(weight, self.gamma)
        loss_dice = self.loss_dice(y_pred, y_true)
        loss_mce = self.loss_mce(y_pred, y_true, weight)
        loss = loss_mce + alpha * loss_dice
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
