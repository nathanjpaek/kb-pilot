import torch
import numpy as np
import torch.nn as nn


class BoundedBeta(torch.distributions.Beta):

    def log_prob(self, x):
        return super().log_prob((x + 1) / 2)


class Beta2(nn.Module):

    def __init__(self, action_dim, init_std=0.25, learn_std=False):
        super(Beta2, self).__init__()
        assert init_std < 0.5, 'Beta distribution has a max std dev of 0.5'
        self.action_dim = action_dim
        self.logstd = nn.Parameter(torch.ones(1, action_dim) * np.log(
            init_std), requires_grad=learn_std)
        self.learn_std = learn_std

    def forward(self, x):
        mean = torch.sigmoid(x)
        var = self.logstd.exp().pow(2)
        """
        alpha = ((1 - mu) / sigma^2 - 1 / mu) * mu^2
        beta  = alpha * (1 / mu - 1)

        Implemented slightly differently for numerical stability.
        """
        alpha = (1 - mean) / var * mean.pow(2) - mean
        beta = (1 - mean) / var * mean - 1 - alpha
        return alpha, beta

    def sample(self, x, deterministic):
        if deterministic is False:
            action = self.evaluate(x).sample()
        else:
            return self.evaluate(x).mean
        return 2 * action - 1

    def evaluate(self, x):
        alpha, beta = self(x)
        return BoundedBeta(alpha, beta)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'action_dim': 4}]
