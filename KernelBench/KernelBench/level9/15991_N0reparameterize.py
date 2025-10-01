import torch
from torch import nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class N0reparameterize(nn.Module):
    """Reparametrize zero mean Gaussian Variable."""

    def __init__(self, input_dim, z_dim, fixed_sigma=None):
        super().__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.sigma_linear = nn.Linear(input_dim, z_dim)
        self.return_means = False
        if fixed_sigma is not None:
            self.register_buffer('fixed_sigma', torch.tensor(fixed_sigma))
        else:
            self.fixed_sigma = None
        self.sigma = None
        self.z = None

    def forward(self, x, n=1):
        if self.fixed_sigma is not None:
            self.sigma = x.new_full((x.shape[0], self.z_dim), self.fixed_sigma)
        else:
            self.sigma = F.softplus(self.sigma_linear(x))
        self.z = self.nsample(n=n)
        return self.z

    def kl(self):
        return -0.5 * torch.sum(1 + 2 * self.sigma.log() - self.sigma ** 2, -1)

    def log_posterior(self):
        return self._log_posterior(self.z)

    def _log_posterior(self, z):
        return Normal(torch.zeros_like(self.sigma), self.sigma).log_prob(z
            ).sum(-1)

    def log_prior(self):
        return Normal(torch.zeros_like(self.sigma), torch.ones_like(self.sigma)
            ).log_prob(self.z).sum(-1)

    def nsample(self, n=1):
        if self.return_means:
            return torch.zeros_like(self.sigma).expand(n, -1, -1)
        eps = Normal(torch.zeros_like(self.sigma), torch.ones_like(self.sigma)
            ).sample((n,))
        return eps * self.sigma

    def deterministic(self):
        """Set to return means."""
        self.return_means = True


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'z_dim': 4}]
