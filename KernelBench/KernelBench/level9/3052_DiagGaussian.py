import torch
import numpy as np
import torch.nn as nn
import torch.utils.data


class BaseDistribution(nn.Module):
    """
    Base distribution of a flow-based model
    Parameters do not depend of target variable (as is the case for a VAE encoder)
    """

    def __init__(self):
        super().__init__()

    def forward(self, num_samples=1):
        """
        Samples from base distribution and calculates log probability
        :param num_samples: Number of samples to draw from the distriubtion
        :return: Samples drawn from the distribution, log probability
        """
        raise NotImplementedError

    def log_prob(self, z):
        """
        Calculate log probability of batch of samples
        :param z: Batch of random variables to determine log probability for
        :return: log probability for each batch element
        """
        raise NotImplementedError


class DiagGaussian(BaseDistribution):
    """
    Multivariate Gaussian distribution with diagonal covariance matrix
    """

    def __init__(self, d):
        """
        Constructor
        :param d: Dimension of Gaussian distribution
        """
        super().__init__()
        self.d = d
        self.loc = nn.Parameter(torch.zeros(1, self.d))
        self.log_scale = nn.Parameter(torch.zeros(1, self.d))

    def forward(self, num_samples=1):
        eps = torch.randn((num_samples, self.d), device=self.loc.device)
        z = self.loc + torch.exp(self.log_scale) * eps
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(self.
            log_scale + 0.5 * torch.pow(eps, 2), 1)
        return z, log_p

    def log_prob(self, z):
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(self.
            log_scale + 0.5 * torch.pow((z - self.loc) / torch.exp(self.
            log_scale), 2), 1)
        return log_p


def get_inputs():
    return []


def get_init_inputs():
    return [[], {'d': 4}]
