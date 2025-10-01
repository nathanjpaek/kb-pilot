import math
import torch
import numpy as np


class SphericalBesselBasis(torch.nn.Module):
    """
    1D spherical Bessel basis

    Parameters
    ----------
    num_radial: int
        Controls maximum frequency.
    cutoff: float
        Cutoff distance in Angstrom.
    """

    def __init__(self, num_radial: 'int', cutoff: 'float'):
        super().__init__()
        self.norm_const = math.sqrt(2 / cutoff ** 3)
        self.frequencies = torch.nn.Parameter(data=torch.tensor(np.pi * np.
            arange(1, num_radial + 1, dtype=np.float32)), requires_grad=True)

    def forward(self, d_scaled):
        return self.norm_const / d_scaled[:, None] * torch.sin(self.
            frequencies * d_scaled[:, None])


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_radial': 4, 'cutoff': 4}]
