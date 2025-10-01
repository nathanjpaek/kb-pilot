import torch
import torch.nn.functional as F
import torch.nn as nn


class Critic(nn.Module):
    """ Neural Network for the Critic Model """

    def __init__(self, state_size, action_size, seed=0, first_layer_units=
        400, second_layer_units=300):
        """Initialize parameters and build model.
				Params
				======
					state_size (int): Dimension of each state
					action_size (int): Dimension of each action
					seed (int): Random seed
					layer1_units (int): Number of nodes in first hidden layer
					layer2_units (int): Number of nodes in second hidden layer
		"""
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(state_size + action_size, first_layer_units)
        self.layer_2 = nn.Linear(first_layer_units, second_layer_units)
        self.layer_3 = nn.Linear(second_layer_units, 1)
        self.layer_4 = nn.Linear(state_size + action_size, first_layer_units)
        self.layer_5 = nn.Linear(first_layer_units, second_layer_units)
        self.layer_6 = nn.Linear(second_layer_units, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        x2 = F.relu(self.layer_4(xu))
        x2 = F.relu(self.layer_5(x2))
        x2 = self.layer_6(x2)
        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        return x1

    def Q2(self, x, u):
        xu = torch.cat([x, u], 1)
        x2 = F.relu(self.layer_4(xu))
        x2 = F.relu(self.layer_5(xu))
        x2 = self.layer_6(xu)
        return x2


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'state_size': 4, 'action_size': 4}]
