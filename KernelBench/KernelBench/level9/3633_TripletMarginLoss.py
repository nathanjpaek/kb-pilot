import torch
import torch.nn as nn


class PairwiseDistance(nn.Module):
    """class for calculating distance

    Arguments:
        nn {[type]} -- [description]
    """

    def __init__(self, smooth=0.0001):
        """Initializer

        Arguments:
            smooth {int} -- [description]
        """
        super(PairwiseDistance, self).__init__()
        self.smooth = smooth

    def forward(self, x1, x2):
        """x1, x2 represent input data

        Arguments:
            x1 {[type]} -- [description]
            x2 {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        assert x1.size() == x2.size()
        diff = torch.abs(x1 - x2)
        out = torch.pow(diff, 2).sum(dim=1)
        return torch.pow(out + self.smooth, 0.5)


class TripletMarginLoss(nn.Module):
    """Triplet loss

    Arguments:
        nn {[type]} -- [description]
    """

    def __init__(self, margin):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin
        self.pdist = PairwiseDistance()

    def forward(self, anchor, positive, negative):
        d_p = self.pdist(anchor, positive)
        d_n = self.pdist(anchor, negative)
        dist_hinge = torch.clamp(self.margin + d_p - d_n, min=0)
        loss = torch.mean(dist_hinge)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'margin': 4}]
