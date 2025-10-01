import torch
import torch.nn as nn
import torch.functional as F
import torch.nn.functional as F


class BoundedBeta(torch.distributions.Beta):

    def log_prob(self, x):
        return super().log_prob((x + 1) / 2)


class Beta(nn.Module):

    def __init__(self, action_dim):
        super(Beta, self).__init__()
        self.action_dim = action_dim

    def forward(self, alpha_beta):
        alpha = 1 + F.softplus(alpha_beta[:, :self.action_dim])
        beta = 1 + F.softplus(alpha_beta[:, self.action_dim:])
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
