import torch
import torch.nn as nn


class DIAYNBaselineModel(nn.Module):
    """The model that computes V(s)"""

    def __init__(self, n_observations, n_hidden, n_policies):
        super().__init__()
        self.linear = nn.Linear(n_observations, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_policies)
        self.n_policies = n_policies

    def forward(self, frame, idx_policy):
        z = torch.tanh(self.linear(frame))
        critic = self.linear2(z)
        critic = critic.reshape(critic.size()[0], self.n_policies, 1)
        critic = critic[torch.arange(critic.size()[0]), idx_policy]
        return critic


def get_inputs():
    return [torch.rand([4, 4]), torch.ones([4], dtype=torch.int64)]


def get_init_inputs():
    return [[], {'n_observations': 4, 'n_hidden': 4, 'n_policies': 4}]
