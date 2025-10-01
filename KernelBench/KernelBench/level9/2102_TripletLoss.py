import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):

    def __init__(self, device, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.device = device
        self.loss = nn.TripletMarginLoss(margin)

    def forward(self, anchor, positive, negative):
        loss = self.loss(anchor, positive, negative)
        return loss

    def distance(self, output1, output2):
        diff = F.pairwise_distance(output1, output2)
        return diff


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'device': 0, 'margin': 4}]
