import torch
import torch.nn as nn
import torch.utils.model_zoo


class pdice_loss(nn.Module):

    def __init__(self, batch=True):
        super(pdice_loss, self).__init__()
        self.batch = batch

    def soft_dice_coeff(self, y_true, y_pred, p):
        smooth = 0.0
        if self.batch:
            pmap = p.clone()
            pmap[pmap >= 0.8] = 1
            pmap[pmap < 0.8] = 0
            y_true_th = y_true * pmap
            y_pred_th = y_pred * pmap
            i = torch.sum(y_true_th)
            j = torch.sum(y_pred_th)
            intersection = torch.sum(y_true_th * y_pred_th)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2.0 * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred, pmap):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred, pmap)
        return loss

    def forward(self, y_pred, y_true, pmap):
        b = self.soft_dice_loss(y_true, y_pred, pmap)
        return b


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
