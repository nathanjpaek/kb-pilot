import torch
from torch import nn
from torch.nn import functional as F


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a postive sample and a negative sample
    """

    def __init__(self):
        super(TripletLoss, self).__init__()

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = F.cosine_similarity(anchor, positive)
        distance_negative = F.cosine_similarity(anchor, negative)
        losses = (1 - distance_positive) ** 2 + (0 - distance_negative) ** 2
        return losses.mean() if size_average else losses.sum()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
