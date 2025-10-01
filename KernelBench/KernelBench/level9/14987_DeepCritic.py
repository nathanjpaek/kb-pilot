import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(fan_in)
    return -lim, lim


class DeepCritic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, device, hidden_size=32):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_size (int): Number of nodes in the network layers

        """
        super(DeepCritic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.device = device
        in_dim = hidden_size + action_size + state_size
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(in_dim, hidden_size)
        self.fc3 = nn.Linear(in_dim, hidden_size)
        self.fc4 = nn.Linear(in_dim, hidden_size)
        self.fc5 = nn.Linear(hidden_size, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc4.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc5.weight.data.uniform_(-0.003, 0.003)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xu = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1(xu))
        x = torch.cat([x, xu], dim=1)
        x = F.relu(self.fc2(x))
        x = torch.cat([x, xu], dim=1)
        x = F.relu(self.fc3(x))
        x = torch.cat([x, xu], dim=1)
        x = F.relu(self.fc4(x))
        return self.fc5(x)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'state_size': 4, 'action_size': 4, 'seed': 4, 'device': 0}]
