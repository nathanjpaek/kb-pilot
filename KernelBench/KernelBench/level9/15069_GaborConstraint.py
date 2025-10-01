import math
import torch
from torch import nn


class GaborConstraint(nn.Module):
    """Constraint mu and sigma, in radians.

    Mu is constrained in [0,pi], sigma s.t full-width at half-maximum of the
    gaussian response is in [1,pi/2]. The full-width at half maximum of the
    Gaussian response is 2*sqrt(2*log(2))/sigma . See Section 2.2 of
    https://arxiv.org/pdf/1711.01161.pdf for more details.
    """

    def __init__(self, kernel_size):
        """Initialize kernel size.

        Args:
        kernel_size: the length of the filter, in samples.
        """
        super(GaborConstraint, self).__init__()
        self._kernel_size = kernel_size

    def forward(self, kernel):
        mu_lower = 0.0
        mu_upper = math.pi
        sigma_lower = 4 * math.sqrt(2 * math.log(2)) / math.pi
        sigma_upper = self._kernel_size * math.sqrt(2 * math.log(2)) / math.pi
        clipped_mu = torch.clamp(kernel[:, 0], mu_lower, mu_upper)
        clipped_sigma = torch.clamp(kernel[:, 1], sigma_lower, sigma_upper)
        return torch.stack([clipped_mu, clipped_sigma], dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'kernel_size': 4}]
