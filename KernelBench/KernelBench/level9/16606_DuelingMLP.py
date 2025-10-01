import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingMLP(nn.Module):

    def __init__(self, state_size, num_actions):
        super().__init__()
        self.linear = nn.Linear(state_size, 256)
        self.value_head = nn.Linear(256, 1)
        self.advantage_head = nn.Linear(256, num_actions)

    def forward(self, x):
        x = x.unsqueeze(0) if len(x.size()) == 1 else x
        x = F.relu(self.linear(x))
        value = self.value_head(x)
        advantage = self.advantage_head(x)
        action_values = (value + (advantage - advantage.mean(dim=1, keepdim
            =True))).squeeze()
        return action_values


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_size': 4, 'num_actions': 4}]
