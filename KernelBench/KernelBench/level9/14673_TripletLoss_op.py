import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss_op(nn.Module):

    def __init__(self, margin=1.0):
        super(TripletLoss_op, self).__init__()
        self.margin = margin

    def forward(self, op, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1).pow(0.5)
        distance_negative = (anchor - negative).pow(2).sum(1).pow(0.5)
        losses = F.relu(op * (distance_positive - distance_negative) + self
            .margin)
        return losses.mean() if size_average else losses.sum()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
