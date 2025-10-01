import torch
import torch.nn as nn
import torch.nn.functional as F


class combLoss(nn.Module):

    def __init__(self, margin, l=1):
        super(combLoss, self).__init__()
        self.margin = margin
        self.l = l

    def forward(self, anchor, pos, neg):
        distance_pos = (anchor - pos).pow(2).sum(1)
        distance_neg = (anchor - neg).pow(2).sum(1)
        distance_cen = (neg - anchor * 0.5 - pos * 0.5).pow(2).sum(1)
        loss = F.relu(distance_pos - self.l * distance_cen + self.margin)
        return loss.mean(), self.triplet_correct(distance_pos, distance_neg)

    def triplet_correct(self, d_pos, d_neg):
        return (d_pos < d_neg).sum()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'margin': 4}]
