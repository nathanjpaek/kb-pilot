import torch
import torch.utils.data
import torch
import torch.nn as nn
from torch.nn import functional as F


class FocalLoss(nn.Module):

    def __init__(self, weight=None, gamma=1.0, num_classes=80):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight
        self.num_classes = num_classes

    def forward(self, input, target):
        CE = F.cross_entropy(input, target, reduction='none')
        p = torch.exp(-CE)
        loss = (1 - p) ** self.gamma * CE
        return loss.sum()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
