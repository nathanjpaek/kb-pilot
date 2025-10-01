import torch
import torch.nn as nn
import torch.nn.functional as F


class tripletLoss(nn.Module):

    def __init__(self, margin):
        super(tripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, pos, neg):
        distance_pos = (anchor - pos).pow(2).sum(1)
        distance_neg = (anchor - neg).pow(2).sum(1)
        loss = F.relu(distance_pos - distance_neg + self.margin)
        return loss.mean(), self.triplet_correct(distance_pos, distance_neg)

    def triplet_correct(self, d_pos, d_neg):
        return (d_pos < d_neg).sum()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'margin': 4}]
