import torch
import numpy as np
import torch.nn as nn


class RBFExpansion(nn.Module):
    """Expand distances between nodes by radial basis functions.

    .. math::
        \\exp(- \\gamma * ||d - \\mu||^2)

    where :math:`d` is the distance between two nodes and :math:`\\mu` helps centralizes
    the distances. We use multiple centers evenly distributed in the range of
    :math:`[\\text{low}, \\text{high}]` with the difference between two adjacent centers
    being :math:`gap`.

    The number of centers is decided by :math:`(\\text{high} - \\text{low}) / \\text{gap}`.
    Choosing fewer centers corresponds to reducing the resolution of the filter.

    Parameters
    ----------
    low : float
        Smallest center. Default to 0.
    high : float
        Largest center. Default to 30.
    gap : float
        Difference between two adjacent centers. :math:`\\gamma` will be computed as the
        reciprocal of gap. Default to 0.1.
    """

    def __init__(self, low=0.0, high=30.0, gap=0.1):
        super(RBFExpansion, self).__init__()
        num_centers = int(np.ceil((high - low) / gap))
        self.centers = np.linspace(low, high, num_centers)
        self.centers = nn.Parameter(torch.tensor(self.centers).float(),
            requires_grad=False)
        self.gamma = 1 / gap

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.centers = nn.Parameter(torch.tensor(self.centers).float(),
            requires_grad=False)

    def forward(self, edge_dists):
        """Expand distances.

        Parameters
        ----------
        edge_dists : float32 tensor of shape (E, 1)
            Distances between end nodes of edges, E for the number of edges.

        Returns
        -------
        float32 tensor of shape (E, len(self.centers))
            Expanded distances.
        """
        radial = edge_dists - self.centers
        coef = -self.gamma
        return torch.exp(coef * radial ** 2)


def get_inputs():
    return [torch.rand([4, 4, 4, 300])]


def get_init_inputs():
    return [[], {}]
