import torch
from torch import nn


class SOSLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, anchors, positives, negatives):
        dist_an = torch.sum(torch.pow(anchors - negatives, 2), dim=1)
        dist_pn = torch.sum(torch.pow(positives - negatives, 2), dim=1)
        nq = anchors.size(dim=0)
        return torch.sum(torch.pow(dist_an - dist_pn, 2)) ** 0.5 / nq


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
