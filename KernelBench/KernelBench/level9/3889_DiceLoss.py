import torch
import torch.utils.data
import torch.nn as nn


class DiceLoss(nn.Module):
    """DICE loss.
    """

    def __init__(self, size_average=True, reduce=True, smooth=100.0, power=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduce = reduce
        self.power = power

    def dice_loss(self, pred, target):
        loss = 0.0
        for index in range(pred.size()[0]):
            iflat = pred[index].view(-1)
            tflat = target[index].view(-1)
            intersection = (iflat * tflat).sum()
            if self.power == 1:
                loss += 1 - (2.0 * intersection + self.smooth) / (iflat.sum
                    () + tflat.sum() + self.smooth)
            else:
                loss += 1 - (2.0 * intersection + self.smooth) / ((iflat **
                    self.power).sum() + (tflat ** self.power).sum() + self.
                    smooth)
        return loss / float(pred.size()[0])

    def dice_loss_batch(self, pred, target):
        iflat = pred.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        if self.power == 1:
            loss = 1 - (2.0 * intersection + self.smooth) / (iflat.sum() +
                tflat.sum() + self.smooth)
        else:
            loss = 1 - (2.0 * intersection + self.smooth) / ((iflat ** self
                .power).sum() + (tflat ** self.power).sum() + self.smooth)
        return loss

    def forward(self, pred, target):
        if not target.size() == pred.size():
            raise ValueError(
                'Target size ({}) must be the same as pred size ({})'.
                format(target.size(), pred.size()))
        if self.reduce:
            loss = self.dice_loss(pred, target)
        else:
            loss = self.dice_loss_batch(pred, target)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
