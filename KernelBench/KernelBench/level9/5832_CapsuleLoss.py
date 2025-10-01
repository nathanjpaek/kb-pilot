import torch
from torch import nn
import torch.nn.functional as F


class CapsuleLoss(nn.Module):

    def __init__(self):
        super(CapsuleLoss, self).__init__()

    def forward(self, output, target):
        class_loss = (target * F.relu(0.9 - output) + 0.5 * (1 - target) *
            F.relu(output - 0.1)).mean()
        return class_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
