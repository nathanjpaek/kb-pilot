import torch
import torch.nn.functional as F
import torch.nn as nn


class ActorNetwork(nn.Module):

    def __init__(self, state_dim, action_dim, seed, fc1_units=256,
        fc2_units=128):
        """ Initialize parameters of model and build its.
        Parameters:
        ===========
        state_dim (int): State space dimension 
        action_dim (int): Action space dimension
        seed (int): Random seed
        fcX_units (int): No. of hidden layers units
        """
        super(ActorNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_dim, fc1_units)
        self.bn1 = nn.LayerNorm(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_dim)
        self.init_parameters()

    def init_parameters(self):
        """ Initialize network weights. """
        self.fc3.weight.data.uniform_(-0.003, 0.003)

    def forward(self, state):
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_dim': 4, 'action_dim': 4, 'seed': 4}]
