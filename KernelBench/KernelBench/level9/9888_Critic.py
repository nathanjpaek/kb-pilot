import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


class Critic(nn.Module):
    """Critic (Value) Model.

    This class construct the model.
    """

    def __init__(self, state_size, action_size, seed, fc1_units=128,
        fc2_units=128, fc3_units=128):
        """ Initialize parameters and build model.

        Args:
            state_size: Integer. Dimension of each state
            action_size: Integer. Dimension of each action
            seed: Integer. Value to set the seed of the model
            fc1_units: Integer. Number of nodes in first fully connect hidden layer
            fc2_units: Integer. Number of nodes in second fully connect hidden layer
            fc3_units: Integer. Number of nodes in third fully connect hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*self.hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*self.hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*self.hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-0.003, 0.003)

    @staticmethod
    def hidden_init(layer):
        fan_in = layer.weight.data.size()[0]
        lim = 1.0 / np.sqrt(fan_in)
        return -lim, lim

    def __repr__(self):
        return 'Critic of Deep Deterministic Policy Gradient Model'

    def __str__(self):
        return 'Critic of Deep Deterministic Policy Gradient Model'

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action)
           pairs -> Q-values.

        Args:
            state: A tensor with the state values
            action: A tensor with the actions values

        Returns:
            A tensor if there is a single output, or a list of tensors if there
                are more than one outputs.
        """
        hidden = F.leaky_relu(self.fc1(state))
        hidden = torch.cat((hidden, action), dim=1)
        hidden = F.leaky_relu(self.fc2(hidden))
        hidden = F.leaky_relu(self.fc3(hidden))
        return self.fc4(hidden)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'state_size': 4, 'action_size': 4, 'seed': 4}]
