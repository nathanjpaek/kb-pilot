import torch
from torch import nn


class ElectronicAsymptotic(nn.Module):
    """Jastrow factor with a correct electronic cusp.

    The Jastrow factor is calculated from distances between all pairs of
    electrons, :math:`d_{ij}`,

    .. math::
        \\mathrm \\gamma
        :=\\sum_{ij}-\\frac{c}{\\alpha(1+\\alpha d_{ij})}

    Args:
        cusp (float): *c*, target cusp value
        alpha (float): :math:`\\alpha`, rate of decay of the cusp function to 1.

    Shape:
        - Input, :math:`d_{ij}`: :math:`(*,N_\\text{pair})`
        - Output, :math:`\\gamma`: :math:`(*)`
    """

    def __init__(self, *, cusp, alpha=1.0):
        super().__init__()
        self.cusp = cusp
        self.alpha = alpha

    def forward(self, dists):
        return -(self.cusp / (self.alpha * (1 + self.alpha * dists))).sum(dim
            =-1)

    def extra_repr(self):
        return f'cusp={self.cusp}, alpha={self.alpha}'


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'cusp': 4}]
