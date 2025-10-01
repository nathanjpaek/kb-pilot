import torch
import torch.nn.functional as F
import torch.nn as nn


class Prediction(nn.Module):
    """Defines the prediction module as an ANN"""

    def __init__(self, state_size: 'int', action_size: 'int', fc1: 'int'=24,
        fc2: 'int'=24):
        super(Prediction, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.fc1 = nn.Linear(state_size, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, action_size)

    def forward(self, state):
        """Prediction unit forward pass. Input state, s_t. Output, action, a_t"""
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class Environment(nn.Module):
    """Defines the Environment module as an ANN"""

    def __init__(self, state_size: 'int', action_size: 'int', fc1: 'int'=24,
        fc2: 'int'=24):
        super(Environment, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.fc1 = nn.Linear(action_size, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, state_size)

    def forward(self, action):
        """Environment unit forward pass. Input action, a_t. Output, s_t+1."""
        x = self.fc1(action)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class DQNetwork(nn.Module):
    """Main DQN network utilizing `Prediction` and `Environment` modules"""

    def __init__(self, state_size: 'int', action_size: 'int'):
        super(DQNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.prediction = Prediction(state_size, action_size)
        self.env = Environment(state_size, action_size)

    def forward(self, state):
        """Returns a_t and s_t+1"""
        action = self.prediction(state)
        predicted_next_state = self.env(action)
        return action, predicted_next_state


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_size': 4, 'action_size': 4}]
