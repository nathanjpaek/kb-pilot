import torch
import torch as t
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F


class A2CActorCont(nn.Module):

    def __init__(self, state_dim, action_dim, action_range):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.mu_head = nn.Linear(16, action_dim)
        self.sigma_head = nn.Linear(16, action_dim)
        self.action_range = action_range

    def forward(self, state, action=None):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        mu = 2.0 * t.tanh(self.mu_head(a))
        sigma = F.softplus(self.sigma_head(a))
        dist = Normal(mu, sigma)
        action = action if action is not None else dist.sample()
        action_entropy = dist.entropy()
        action = action.clamp(-self.action_range, self.action_range)
        action_log_prob = dist.log_prob(action)
        return action, action_log_prob, action_entropy


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_dim': 4, 'action_dim': 4, 'action_range': 4}]
