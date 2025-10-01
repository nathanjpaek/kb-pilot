import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model. Deep Net function approximator for q(s,a)"""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Parameters:
        ==========
            state_size (int): This is the dimension of each state.
            action_size (int): This is the dimension of each action.
            seed (int): This gives the random seed.
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, action_size)

    def forward(self, state):
        """This builds a network that maps a state to action values."""
        state = self.fc1(state)
        state = F.relu(state)
        state = self.fc2(state)
        state = F.relu(state)
        state = self.fc3(state)
        state = F.relu(state)
        state = self.fc4(state)
        return state


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_size': 4, 'action_size': 4, 'seed': 4}]
