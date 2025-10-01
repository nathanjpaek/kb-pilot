import torch
import torch.nn as nn


class DuelingNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed):
        super(DuelingNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.action_size = action_size
        self.fc1 = nn.Linear(state_size, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.relu2 = nn.ReLU()
        self.fc3_to_state_value = nn.Linear(64, 1)
        self.fc3_to_action_value = nn.Linear(64, self.action_size)

    def forward(self, state):
        x = self.fc1(state)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        v_x = self.fc3_to_state_value(x)
        a_x = self.fc3_to_action_value(x)
        average_operator = 1 / self.action_size * a_x
        x = v_x + (a_x - average_operator)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_size': 4, 'action_size': 4, 'seed': 4}]
