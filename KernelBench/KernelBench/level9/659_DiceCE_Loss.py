import torch
from torch import nn
from torch.nn import functional as F
from torch import sigmoid


class DiceCE_Loss(nn.Module):
    """
    Taken from https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
    """

    def __init__(self, weight=None, size_average=True):
        super(DiceCE_Loss, self).__init__()

    def forward(self, out, targets, smooth=1e-05):
        BCE = F.binary_cross_entropy_with_logits(out, targets, reduction='mean'
            )
        out = sigmoid(out)
        num = targets.size(0)
        out = out.view(num, -1)
        targets = targets.view(num, -1)
        intersection = out * targets
        dice = (2.0 * intersection.sum(1) + smooth) / (out.sum(1) + targets
            .sum(1) + smooth)
        dice_loss = dice.sum() / num
        Dice_BCE = 0.5 * BCE - dice_loss
        return Dice_BCE


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
