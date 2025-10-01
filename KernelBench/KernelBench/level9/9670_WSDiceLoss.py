import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.parallel


class WSDiceLoss(nn.Module):

    def __init__(self, smooth=100.0, power=2.0, v2=0.85, v1=0.15):
        super().__init__()
        self.smooth = smooth
        self.power = power
        self.v2 = v2
        self.v1 = v1

    def dice_loss(self, pred, target):
        iflat = pred.reshape(pred.shape[0], -1)
        tflat = target.reshape(pred.shape[0], -1)
        wt = tflat * (self.v2 - self.v1) + self.v1
        g_pred = wt * (2 * iflat - 1)
        g = wt * (2 * tflat - 1)
        intersection = (g_pred * g).sum(-1)
        loss = 1 - (2.0 * intersection + self.smooth) / ((g_pred ** self.
            power).sum(-1) + (g ** self.power).sum(-1) + self.smooth)
        return loss.mean()

    def forward(self, pred, target, weight_mask=None):
        loss = self.dice_loss(pred, target)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
