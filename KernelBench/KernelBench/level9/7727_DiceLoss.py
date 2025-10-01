import torch
from torch import nn
from torch.nn import functional as F


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, output, target):
        prediction = F.sigmoid(output)
        intersection = torch.sum(prediction * target)
        union = torch.sum(prediction) + torch.sum(target) + 1e-07
        return 1 - 2 * intersection / union


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
