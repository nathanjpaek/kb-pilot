import torch
import torch.nn as nn


class MaskedHuberLoss(torch.nn.Module):

    def __init__(self):
        super(MaskedHuberLoss, self).__init__()

    def forward(self, output, labels, mask):
        lossHuber = nn.SmoothL1Loss(reduction='none')
        l = lossHuber(output * mask, labels * mask)
        l = l.sum(dim=(1, 2))
        mask = mask.sum(dim=(1, 2))
        l = l / mask
        return l.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
