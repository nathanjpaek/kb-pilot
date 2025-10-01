import torch
import torch.nn as nn


class TripletLoss(torch.nn.Module):
    """
    Triplet loss function.
    """

    def __init__(self, margin=2.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        squarred_distance_1 = (anchor - positive).pow(2).sum(1)
        squarred_distance_2 = (anchor - negative).pow(2).sum(1)
        triplet_loss = nn.ReLU()(self.margin + squarred_distance_1 -
            squarred_distance_2).mean()
        return triplet_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
