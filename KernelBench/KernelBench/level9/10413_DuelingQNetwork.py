import torch
import torch.nn as nn
from collections import OrderedDict


class DuelingQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_advantage=[512,
        512], hidden_state_value=[512, 512]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_layers (list): List containing the hidden layer sizes
        """
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        hidden_layers = [state_size] + hidden_advantage
        advantage_layers = OrderedDict()
        for idx, (hl_in, hl_out) in enumerate(zip(hidden_layers[:-1],
            hidden_layers[1:])):
            advantage_layers['adv_fc_' + str(idx)] = nn.Linear(hl_in, hl_out)
            advantage_layers['adv_activation_' + str(idx)] = nn.ReLU()
        advantage_layers['adv_output'] = nn.Linear(hidden_layers[-1],
            action_size)
        self.network_advantage = nn.Sequential(advantage_layers)
        value_layers = OrderedDict()
        hidden_layers = [state_size] + hidden_state_value
        for idx, (hl_in, hl_out) in enumerate(zip(hidden_layers[:-1],
            hidden_layers[1:])):
            value_layers['val_fc_' + str(idx)] = nn.Linear(hl_in, hl_out)
            value_layers['val_activation_' + str(idx)] = nn.ReLU()
        value_layers['val_output'] = nn.Linear(hidden_layers[-1], 1)
        self.network_value = nn.Sequential(value_layers)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        advantage = self.network_advantage(state)
        value = self.network_value(state)
        return advantage.sub_(advantage.mean()).add_(value)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_size': 4, 'action_size': 4, 'seed': 4}]
