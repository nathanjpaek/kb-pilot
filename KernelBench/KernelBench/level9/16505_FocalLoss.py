import torch
import numpy as np
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
        prior = np.array([0.119217, 0.15927, 0.570566, 0.1045, 0.04089, 
            0.005522])
        self.prior = torch.tensor(prior).float()
        self.weight_b = torch.from_numpy(np.array([1.11, 1.06, 1.01, 1.16, 
            1.84, 10.0, 1.0])).float()

    def forward(self, input, target):
        CE = F.cross_entropy(input, target, reduction='none')
        p = torch.exp(-CE)
        loss = (1 - p) ** self.gamma * CE
        return loss.sum() / CE.shape[0]


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
