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


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
