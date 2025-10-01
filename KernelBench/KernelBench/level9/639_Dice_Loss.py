import torch
from torch import nn
from torch import sigmoid


class Dice_Loss(nn.Module):
    """
    Taken from https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
    """

    def __init__(self, weight=None, size_average=True):
        super(Dice_Loss, self).__init__()

    def forward(self, out, targets, smooth=1):
        out = sigmoid(out)
        out = out.view(-1)
        targets = targets.view(-1)
        intersection = (out * targets).sum()
        dice = (2.0 * intersection + smooth) / (out.sum() + targets.sum() +
            smooth)
        return 1 - dice


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
