import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.backends.cudnn
import torch.utils.data


def dice_loss(preds, trues, weight=None, is_average=True):
    num = preds.size(0)
    preds = preds.view(num, -1)
    trues = trues.view(num, -1)
    if weight is not None:
        w = torch.autograd.Variable(weight).view(num, -1)
        preds = preds * w
        trues = trues * w
    intersection = (preds * trues).sum(1)
    scores = 2.0 * (intersection + 1) / (preds.sum(1) + trues.sum(1) + 1)
    if is_average:
        score = scores.sum() / num
        return torch.clamp(score, 0.0, 1.0)
    else:
        return scores


class DiceLoss(nn.Module):

    def __init__(self, size_average=True):
        super().__init__()
        self.size_average = size_average

    def forward(self, input, target, weight=None):
        return 1 - dice_loss(F.sigmoid(input), target, weight=weight,
            is_average=self.size_average)


class BCEDiceLoss(nn.Module):

    def __init__(self, size_average=True):
        super().__init__()
        self.size_average = size_average
        self.dice = DiceLoss(size_average=size_average)

    def forward(self, input, target, weight=None):
        return nn.modules.loss.BCEWithLogitsLoss(size_average=self.
            size_average, weight=weight)(input, target) + self.dice(input,
            target, weight=weight)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
