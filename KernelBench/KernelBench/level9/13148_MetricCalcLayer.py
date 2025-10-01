import torch
import torch.nn as nn


class MetricCalcLayer(nn.Module):
    """
    Description
    -----------
    Calculate metric in equation 3 of paper.

    Parameters
    ----------
    nhid : int
        The dimension of mapped features in the graph generating procedure.
    """

    def __init__(self, nhid):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, nhid))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, h):
        """
        Parameters
        ----------
        h : tensor
            The result of the Hadamard product in equation 3 of paper.
        """
        return h * self.weight


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nhid': 4}]
