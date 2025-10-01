import torch
import torch.nn.functional
import torch.nn as nn


def centercrop(image, w, h):
    _nt, _ct, ht, wt = image.size()
    padw, padh = (wt - w) // 2, (ht - h) // 2
    if padw > 0 and padh > 0:
        image = image[:, :, padh:-padh, padw:-padw]
    return image


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


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
