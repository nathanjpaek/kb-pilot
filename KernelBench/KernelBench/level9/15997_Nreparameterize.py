import torch
from torch import nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class Nreparameterize(nn.Module):
    """Reparametrize Gaussian variable."""

    def __init__(self, input_dim, z_dim):
        super().__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.sigma_linear = nn.Linear(input_dim, z_dim)
        self.mu_linear = nn.Linear(input_dim, z_dim)
        self.return_means = False
        self.mu, self.sigma, self.z = None, None, None

    def forward(self, x, n=1):
        self.mu = self.mu_linear(x)
        self.sigma = F.softplus(self.sigma_linear(x))
        self.z = self.nsample(n=n)
        return self.z

    def kl(self):
        return -0.5 * torch.sum(1 + 2 * self.sigma.log() - self.mu.pow(2) -
            self.sigma ** 2, -1)

    def log_posterior(self):
        return self._log_posterior(self.z)

    def _log_posterior(self, z):
        return Normal(self.mu, self.sigma).log_prob(z).sum(-1)

    def log_prior(self):
        return Normal(torch.zeros_like(self.mu), torch.ones_like(self.sigma)
            ).log_prob(self.z).sum(-1)

    def nsample(self, n=1):
        if self.return_means:
            return self.mu.expand(n, -1, -1)
        eps = Normal(torch.zeros_like(self.mu), torch.ones_like(self.mu)
            ).sample((n,))
        return self.mu + eps * self.sigma

    def deterministic(self):
        """Set to return means."""
        self.return_means = True


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'z_dim': 4}]
