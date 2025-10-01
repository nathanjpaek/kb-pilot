import torch
import torch.nn.functional as F
from torch import nn


class DQN(nn.Module):

    def __init__(self, observation_size, action_size, H1=200, H2=160, H3=
        120, H4=60):
        """

        :param observation_size: Size of belief as defined in belief_agent.py
        :param action_size: Model has 1 output for every single possible card in the deck.
        :param H1: size of hidden layer 1
        :param H2: size of hidden layer 2
        """
        super().__init__()
        self.fc1 = torch.nn.Linear(observation_size, H1)
        self.fc2 = torch.nn.Linear(H1, H2)
        self.fc3 = torch.nn.Linear(H2, H3)
        self.fc4 = torch.nn.Linear(H3, H4)
        self.fc5 = torch.nn.Linear(H4, action_size)

    def forward(self, observation):
        """
        Maps observation to action values.
        """
        x = F.relu(self.fc1(observation))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'observation_size': 4, 'action_size': 4}]
