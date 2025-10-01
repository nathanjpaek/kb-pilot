import torch
import torch.nn as nn
import torch.nn.functional as F


class CriticNet(nn.Module):
    """Critic (Value estimator) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64,
        fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super().__init__()
        None
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)

    def forward(self, state):
        """Build a network that maps state -> expected Q values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        v = self.fc3(x)
        return v.squeeze()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_size': 4, 'action_size': 4, 'seed': 4}]
