import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianPolicyFunction(nn.Module):
    """fully connected 200x200 hidden layers"""

    def __init__(self, state_dim, action_dim):
        super(GaussianPolicyFunction, self).__init__()
        self.fc1 = nn.Linear(state_dim, 200)
        self.fc2 = nn.Linear(200, 200)
        self.mu_out = nn.Linear(200, action_dim)
        self.sigma_out = nn.Linear(200, action_dim)

    def forward(self, x):
        """return: action between [-1,1]"""
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        return torch.tanh(self.mu_out(x)), F.softplus(self.sigma_out(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_dim': 4, 'action_dim': 4}]
