import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNet(nn.Module):

    def __init__(self, num_state, num_action):
        super(ActorNet, self).__init__()
        self.num_state = num_state
        self.num_action = num_action
        self.fc1 = nn.Linear(self.num_state, 100)
        self.fc2 = nn.Linear(100, 100)
        self.mu_head = nn.Linear(100, self.num_action)
        self.sigma_head = nn.Linear(100, self.num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = 50.0 * F.tanh(self.mu_head(x))
        sigma = 0.05 * F.softplus(self.sigma_head(x))
        return mu, sigma


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_state': 4, 'num_action': 4}]
