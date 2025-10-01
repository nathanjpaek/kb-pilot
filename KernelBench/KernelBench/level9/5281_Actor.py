import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):

    def __init__(self, state_dim: 'int', action_dim: 'int'):
        """
        Initialize the network
        param: state_dim : Size of the state space
        param: action_dim: Size of the action space
        """
        super(Actor, self).__init__()
        hidden_dim_1 = 256
        hidden_dim_2 = 256
        self.fc1 = nn.Linear(state_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc4 = nn.Linear(hidden_dim_2, action_dim)
        self.fc1.weight.data.uniform_(-1 / np.sqrt(state_dim), 1 / np.sqrt(
            state_dim))
        self.fc2.weight.data.uniform_(-1 / np.sqrt(hidden_dim_1), 1 / np.
            sqrt(hidden_dim_1))
        self.fc4.weight.data.uniform_(-0.003, 0.003)

    def forward(self, state: 'torch.Tensor') ->torch.Tensor:
        """
        Define the forward pass
        param: state: The state of the environment
        """
        x = state
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = torch.tanh(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_dim': 4, 'action_dim': 4}]
