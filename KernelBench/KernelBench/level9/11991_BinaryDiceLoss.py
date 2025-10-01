import torch
import torch.nn as nn


class BinaryDiceLoss(nn.Module):

    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        N = targets.size()[0]
        smooth = 1
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)
        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1
            ) + targets_flat.sum(1) + smooth)
        loss = 1 - N_dice_eff.sum() / N
        return loss

    def dice(self, prec, label):
        smooth = 1
        input_flat = prec.view(1, -1)
        targets_flat = label.view(1, -1)
        intersection = input_flat * targets_flat
        d = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) +
            targets_flat.sum(1) + smooth)
        return d.sum()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
