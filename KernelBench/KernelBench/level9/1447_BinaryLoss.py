import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryLoss(nn.Module):
    """
    Computes contrastive loss[1, 2] twice, one time for the distance between query and positive example,
        and another for the distance between query and negative example. Both use l2-distance.
    [1] http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf, equation 4
    [2] https://gist.github.com/harveyslash/725fcc68df112980328951b3426c0e0b#file-contrastive-loss-py
    """

    def __init__(self, margin=1.0):
        """
        Args:
            margin: margin (float, optional): Default: `1.0`.
        """
        super(BinaryLoss, self).__init__()
        self.margin = margin

    def forward(self, query, positive, negative):
        distance_positive = F.pairwise_distance(query, positive)
        distance_negative = F.pairwise_distance(query, negative)
        return torch.pow(distance_positive, 2) + torch.pow(torch.clamp(self
            .margin - distance_negative, min=0.0), 2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
