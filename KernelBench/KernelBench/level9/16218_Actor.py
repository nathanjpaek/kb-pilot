import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """Initialize parameters and build model.
       An nn.Module contains layers, and a method
       forward(input)that returns the output.
       Weights (learnable params) are inherently defined here.

        Args:
            state_dim (int): Dimension of each state
            action_dim (int): Dimension of each action
            max_action (float): highest action to take

        Return:
            action output of network with tanh activation
    """

    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.fc1(state))
        a = F.relu(self.fc2(a))
        return self.max_action * torch.tanh(self.fc3(a))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_dim': 4, 'action_dim': 4, 'max_action': 4}]
