import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNPolicy(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, conv1_filters=32,
        conv2_filters=32, conv3_filters=32, fc1_units=200, fc2_units=200):
        """Initialize parameters and build model.
        Params
        ======
            state_size (list): Shape of each state image, e.g [3, 28, 28] 
            action_size (int): Dimension of each action
            seed (int): Random seed
            conv1_filters (int): Number of filters for first CNN layer
            conv2_filters (int): Number of filters for second CNN layer
            conv3_filters (int): Number of filters for third CNN layer
            fc1_units (int): Number of nodes in first FC layer
            fc2_units (int): Number of nodes in second FC layer
        """
        super(CNNPolicy, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(state_size[0], conv1_filters, 3, padding=1)
        self.conv2 = nn.Conv2d(conv1_filters, conv2_filters, 3, padding=1)
        self.conv3 = nn.Conv2d(conv2_filters, conv3_filters, 3, padding=1)
        self.fc1 = nn.Linear(conv3_filters * state_size[1] * state_size[2],
            fc1_units)
        self.drop = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_size': [4, 4, 4], 'action_size': 4, 'seed': 4}]
