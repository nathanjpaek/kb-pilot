import torch
import torch.nn.functional as F
import torch.nn as nn


class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed, num_layers=1,
        hidden_size=64):
        """
        Initialize parameters and build model.

        parameters:
            state_size : (int) Dimension of each state
            action_size : (int) Dimension of each action
            seed : (int) Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.input_layer = nn.Linear(self.state_size, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(self.hidden_size,
            self.hidden_size) for i in range(num_layers - 1)])
        self.final = nn.Linear(self.hidden_size, self.action_size)

    def forward(self, state):
        """
        Returns q-values for a given state

        parameters:
            state : (np.Array) the state the agent is in
        returns:
            action_values : (np.Array)
        """
        x = self.input_layer(state)
        x = F.relu(x)
        for i, l in enumerate(self.hidden_layers):
            x = l(x)
            x = F.relu(x)
        action_values = self.final(x)
        return action_values


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_size': 4, 'action_size': 4, 'seed': 4}]
