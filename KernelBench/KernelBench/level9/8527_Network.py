import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):

    def __init__(self, input_size, number_of_actions):
        super(Network, self).__init__()
        self.input_size = input_size
        self.number_of_actions = number_of_actions
        self.full_connection1 = nn.Linear(input_size, 30)
        self.full_connection2 = nn.Linear(30, number_of_actions)

    def forward(self, state):
        hidden_neurons = F.relu(self.full_connection1(state))
        q_values = self.full_connection2(hidden_neurons)
        return q_values


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'number_of_actions': 4}]
