import torch
import torch.distributions as dist
import torch.nn as nn


class Decoder_h(nn.Module):

    def __init__(self, B, H_dim):
        super().__init__()
        self.B = B
        self.H_dim = H_dim
        self.make_parameters()

    def make_parameters(self):
        self.mu = nn.Linear(self.H_dim, self.B, bias=False)
        self.sigma = nn.Linear(self.H_dim, self.B, bias=False)
        torch.nn.init.uniform_(self.sigma.weight, a=1.0, b=2.0)

    def _log_likelihood(self, h):
        """
        h: shape=(BS,N,H_dim)
        """
        BS, S, H_dim = h.shape
        return dist.Normal(self.mu.weight.view(1, 1, self.B, H_dim), self.
            sigma.weight.view(1, 1, self.B, self.H_dim)).log_prob(h.view(BS,
            S, 1, H_dim))

    def forward(self, z):
        """
        z: shape = (BS,N) or (BS,) or (1,)
        """
        h_dist = dist.Normal(self.mu.weight[z], self.sigma.weight[z])
        return h_dist.rsample()


def get_inputs():
    return [torch.ones([4], dtype=torch.int64)]


def get_init_inputs():
    return [[], {'B': 4, 'H_dim': 4}]
