import torch
import torch.nn.functional as F
from torch import nn


class PlanarNormalizingFlow(nn.Module):
    """
    Planar normalizing flow [Rezende & Mohamed 2015].
    Provides a tighter bound on the ELBO by giving more expressive
    power to the approximate distribution, such as by introducing
    covariance between terms.
    """

    def __init__(self, in_features):
        super(PlanarNormalizingFlow, self).__init__()
        self.u = nn.Parameter(torch.randn(in_features))
        self.w = nn.Parameter(torch.randn(in_features))
        self.b = nn.Parameter(torch.ones(1))

    def forward(self, z):
        uw = torch.dot(self.u, self.w)
        muw = -1 + F.softplus(uw)
        uhat = self.u + (muw - uw) * torch.transpose(self.w, 0, -1
            ) / torch.sum(self.w ** 2)
        zwb = torch.mv(z, self.w) + self.b
        f_z = z + uhat.view(1, -1) * F.tanh(zwb).view(-1, 1)
        psi = (1 - F.tanh(zwb) ** 2).view(-1, 1) * self.w.view(1, -1)
        psi_u = torch.mv(psi, uhat)
        logdet_jacobian = torch.log(torch.abs(1 + psi_u) + 1e-08)
        return f_z, logdet_jacobian


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4}]
