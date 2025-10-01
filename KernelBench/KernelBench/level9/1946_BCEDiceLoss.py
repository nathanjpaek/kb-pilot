import torch
import torch.utils.data
import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        bce = F.binary_cross_entropy_with_logits(output, target)
        smooth = 1e-05
        output = torch.sigmoid(output)
        num = target.size(0)
        output = output.view(num, -1)
        target = target.view(num, -1)
        intersection = output * target
        dice = (2.0 * intersection.sum(1) + smooth) / (output.sum(1) +
            target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
